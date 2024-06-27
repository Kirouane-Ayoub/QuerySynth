# QuerySynth

## Overview
```

 ██████╗ ██╗   ██╗███████╗██████╗ ██╗   ██╗███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗
██╔═══██╗██║   ██║██╔════╝██╔══██╗╚██╗ ██╔╝██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║
██║   ██║██║   ██║█████╗  ██████╔╝ ╚████╔╝ ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║
██║▄▄ ██║██║   ██║██╔══╝  ██╔══██╗  ╚██╔╝  ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║
╚██████╔╝╚██████╔╝███████╗██║  ██║   ██║   ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║
 ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝
                                         LlamaIndex based 🦙                                              
                                                                                    
```
**QuerySynth** is a sophisticated information retrieval chat system designed to provide accurate and insightful responses to user queries. Leveraging state-of-the-art machine learning models and a hybrid search approach, **QuerySynth** is capable of handling both general questions and complex queries by breaking them down into manageable parts. The project integrates multiple advanced technologies to deliver a seamless user experience.

## Features

- **Individual Query Engine Tools**: Designed for answering general questions directly from the data sources.
- **Sub Query Engine Tool**: Handles complex queries by decomposing them into sub-questions, retrieving intermediate responses, and synthesizing a final comprehensive answer.
- **Command-R-Plus by Cohere**: Utilized as the main language model to understand and generate responses.
- **Embed-English-v3.0 by Cohere**: Used as the embedding model for semantic understanding and vector representation of the data.
- **Weaviate**: Acts as the vector store database to efficiently manage and query vector embeddings.
- **Hybrid Search**: Combines vector-based search with traditional keyword search to enhance the accuracy and relevance of search results.
- **Mesop UI**: Provides a user-friendly and responsive interface for interacting with the system.

## Data Source

The primary data source for **QuerySynth** is a folder containing Txt / pdf documents. These documents are indexed and queried to extract relevant information in response to user queries.

## Installation

### Prerequisites

- Python 3.10 or later
- Virtual environment tool (venv)

### Steps

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Kirouane-Ayoub/QuerySynth.git
    cd QuerySynth
    ```

2. **Create a virtual environment**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up Weaviate**:

    Follow the instructions on the [Weaviate documentation](https://weaviate.io/developers/weaviate) to set up your Weaviate instance.

5. **Configure environment variables**:

    Create a `.env` file in the root directory and add your configuration settings (e.g., API keys, database URLs).

6. **Run the application**:

    ```bash
    mesop src/app.py
    ```

## Usage

Once the application is running, open your web browser and navigate to the provided local URL (e.g., `http://localhost:32123`) to access the Mesop UI. From here, you can enter queries and receive responses based on the indexed documents.

