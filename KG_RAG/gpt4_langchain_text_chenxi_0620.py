#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/tomasonjo/blogs/blob/master/llm/enhancing_rag_with_graph.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


get_ipython().system('pip install --upgrade --quiet  langchain langchain-community langchain-openai langchain-experimental neo4j wikipedia tiktoken yfiles_jupyter_graphs')


# In[ ]:


from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough

try:
  import google.colab
  from google.colab import output
  output.enable_custom_widget_manager()
except:
  pass


# In[ ]:


os.environ["OPENAI_API_KEY"] = ""
os.environ["NEO4J_URI"] = "" #Neo4j Aura free instance
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = ""

#graph = Neo4jGraph()


# In[ ]:


get_ipython().system('pip install langchain-neo4j')


# In[ ]:


from langchain_neo4j import Neo4jGraph


# In[ ]:


graph = Neo4jGraph()


# In[ ]:


# Install package, compatible with API partitioning
get_ipython().system('pip install --upgrade --quiet langchain-unstructured unstructured-client unstructured "unstructured[pdf]" python-magic')


# In[ ]:


from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

file_paths = [
    #"./example_data/layout-parser-paper.pdf",
    #"./data/Pharmacology_Katzung.txt",
    "./data/InternalMed_Harrison.txt",
]

loader = UnstructuredLoader(
    file_paths,
    #chunking_strategy="basic",
    #max_characters=1000000,
    #include_orig_elements=False,
    )

raw_documents = loader.load()

print("Number of LangChain documents:", len(raw_documents))
print("Length of text in the document:", len(raw_documents[0].page_content))


# In[ ]:


# Need to explore more based on our datasets
# chunking strategy

text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents[-100:])


# In[ ]:


# use LLMGraphTransformer in langchain to simply construct a knowledge graph and store in neo4j graph database
# have to choose small portion of documents due to the speed...

llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
llm_transformer = LLMGraphTransformer(llm=llm)

graph_documents = llm_transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)


# In[ ]:


# explore to see the construted graph

# directly show the graph resulting from the given Cypher query
default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"

def showGraph(cypher: str = default_cypher):
    # create a neo4j session to run queries
    driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],
        auth = (os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"]))
    session = driver.session()
    widget = GraphWidget(graph = session.run(cypher).graph())
    widget.node_label_mapping = 'id'
    #display(widget)
    return widget

showGraph()


# In[ ]:


# Unstructured data retriever
# use the Neo4jVector.from_existing_graph method to add both keyword and vector retrieval to documents for a hybrid search approach

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)


# In[ ]:


# Graph retriever using LangChain Expression Language (LCEL)

graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the drug, disease, or diagnose entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting drug and disease entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)


# In[ ]:


# random test:

entity_chain.invoke({"question": "How do you diagnose and manage latent tuberculosis infection (LTBI)?"}).names


# In[ ]:


# detect entities in the question

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL(node) {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

# The structured_retriever function starts by detecting entities in the user question.
# Next, it iterates over the detected entities and uses a Cypher template to retrieve the neighborhood of relevant nodes


# In[ ]:


print(structured_retriever("How do you diagnose and manage latent tuberculosis infection (LTBI)?"))

# which is not in the current database


# In[ ]:


print(structured_retriever("What do some tumor markers indicate?"))


# In[ ]:


# combine the unstructured and graph retriever to create the final context that will be passed to an LLM.

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data


# In[ ]:


# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)


# In[ ]:


template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)


# In[ ]:


chain.invoke({"question": "How do you diagnose and manage latent tuberculosis infection (LTBI)?"})


# In[ ]:


# test a follow up question

chain.invoke(
    {
        "question": "How do you evaluate a patient with chronic cough lasting >8 weeks?",
        "chat_history": [("How do you diagnose and manage latent tuberculosis infection (LTBI)?", "The provided context does not contain specific information on diagnosing and managing latent tuberculosis infection (LTBI). For LTBI, diagnosis typically involves a tuberculin skin test (TST) or an interferon-gamma release assay (IGRA). Management usually includes preventive treatment with medications such as isoniazid or rifampin to reduce the risk of progression to active tuberculosis. For detailed guidance, consult relevant clinical guidelines or a healthcare professional.")],
    }
)


# In[ ]:




