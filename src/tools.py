import os

import weaviate
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from models import service_context

load_dotenv()

# Get environment variables for Weaviate cluster URL and API key

cluster_url = os.getenv("WCD_DEMO_URL")
api_key = os.getenv("WCD_DEMO_RO_KEY")

# Connect to Weaviate instance using WCS (Weaviate Cloud Service)

print("--------------Connecting to weaviate Cluster --------------")
client = weaviate.connect_to_wcs(
    cluster_url=cluster_url,
    auth_credentials=weaviate.auth.AuthApiKey(api_key),
)
print("--------------Connected --------------")

# Connect to local instance
# client = weaviate.connect_to_local()


# Reads documents from the specified directory ('src/input_data')
print("--------------Load Documents --------------")

documents = SimpleDirectoryReader(input_dir="src/input_data").load_data()

print("-------------- Documents Loaded --------------")

# Create a vector store using Weaviate as the backend
print("-------------- Create a vector store --------------")

vector_store = WeaviateVectorStore(weaviate_client=client)

# Create a storage context with the vector store
print("-------------- storage context --------------")

storage_context = StorageContext.from_defaults(vector_store=vector_store)


# Create an index from the loaded documents
print("-------------- Create an index --------------")

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
).as_query_engine(
    vector_store_query_mode="hybrid",
    similarity_top_k=4,
    alpha=0.5,
)

# Create individual query engine tools

print("-------------- Create individual_query_engine_tools --------------")

individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index,
        metadata=ToolMetadata(
            name="query_engine",
            description="useful for when you want to answer general questions.",
        ),
    )
]

# Create a sub-query engine for handling more complex queries
print("-------------- Create sub_query_engine_tool --------------")

sub_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools, service_context=service_context
)

print("-------------- Create tool for the sub-query engine --------------")

# Create a tool for the sub-query engine

sub_query_engine_tool = QueryEngineTool(
    query_engine=sub_query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description=(
            "useful for when you want to answer queries that require analyzing"
        ),
    ),
)
print(
    "-------------- Combine the individual query engine tools and the sub-query engine tool --------------"
)

# Combine the individual query engine tools and the sub-query engine tool

tools = individual_query_engine_tools + [sub_query_engine_tool]
