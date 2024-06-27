import os

import settings
from dotenv import load_dotenv
from llama_index.core import ServiceContext
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere

load_dotenv()

cohere_api_key = os.getenv("CO_API_KEY")

# Initialize the embedding model with the specified model name and input type 'search_query'
embed_model = CohereEmbedding(
    model_name=settings.COHERE_EMBEDDING_MODEL,
    input_type="search_query",
    api_key=cohere_api_key,
)
# Initialize the language model with the specified model name

llm = Cohere(model=settings.COHERE_LLM_MODEL, api_key=cohere_api_key)

# Create the service context with the cohere model for generation and embedding model
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
