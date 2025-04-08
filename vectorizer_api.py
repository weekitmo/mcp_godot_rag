#!/usr/bin/env python3

import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import chromadb
from datetime import datetime
import openai
import time
import dotenv

client: openai.OpenAI

base_url = "https://api.siliconflow.cn/v1"


class ChunkVectorizerAPI:
    """Generate embeddings from text chunks using the OpenAI API and store them in a ChromaDB vector database."""

    def __init__(
        self,
        input_file: str,
        db_directory: str,
        api_key: str | None = None,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the vectorizer with input path and API parameters.

        Args:
            input_file: Path to the input JSONL file containing text chunks
            db_directory: Directory where ChromaDB will store the vector database
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
            model_name: The name of the embedding model to use
            batch_size: Batch size for embedding generation
            max_retries: Maximum number of retries for API calls
            retry_delay: Initial delay between retries (in seconds)
        """
        self.input_file = input_file
        self.db_directory = db_directory
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        global client
        _api_key = ""

        # Set up OpenAI API
        if api_key:
            _api_key = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            api_key_env = os.environ.get("OPENAI_API_KEY")
            if api_key_env is not None:
                _api_key = api_key_env
            else:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is set but has None value"
                )
        else:
            # Try to load from .env.local
            env_file = ".env.local"
            if os.path.exists(env_file):
                dotenv.load_dotenv(env_file)
                api_key_env = os.environ.get("OPENAI_API_KEY")
                if api_key_env is not None:
                    _api_key = api_key_env
                    print(f"Loaded API key from {env_file}")
                else:
                    raise ValueError(f"OPENAI_API_KEY not found in {env_file}")
            else:
                raise ValueError(
                    "OpenAI API key must be provided either as an argument, environment variable, or in .env.local file"
                )
        client = openai.Client(base_url=base_url, api_key=_api_key)

        # Create collection name
        collection_base_name = os.path.basename(input_file).replace(".jsonl", "")
        model_short_name = model_name.replace("/", "-")
        collection_name = f"{collection_base_name}_{model_short_name}"

        # Truncate if longer than 63 chars (ChromaDB limitation)
        if len(collection_name) > 63:
            collection_name = collection_name[:63]

        self.collection_name = collection_name
        print(
            f"Collection name: {self.collection_name} ({len(self.collection_name)} chars)"
        )

        # Initialize the ChromaDB client
        print(f"Initializing ChromaDB at {db_directory}")
        self.client = chromadb.PersistentClient(path=db_directory)

        # Delete the collection if it exists
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except Exception as e:
            print(f"No existing collection to delete: {e}")

        # Create embedding function using OpenAI API
        self.embedding_function = OpenAIEmbeddingFunction(
            model_name=self.model_name,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

        # Create a new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,  # type: ignore
            metadata={
                "hnsw:space": "cosine"
            },  # Using cosine similarity for semantic search
        )
        print(f"Created new collection: {self.collection_name}")

        # Append the collection name to artifacts/collections.txt
        collections_file = "artifacts/vector_stores/collections.txt"
        os.makedirs(os.path.dirname(collections_file), exist_ok=True)
        with open(collections_file, "a+", encoding="utf-8") as f:
            f.seek(0)  # Move to the beginning of the file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{self.collection_name} ({timestamp}) - API model: {model_name}\n")

    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from the input JSONL file."""
        chunks = []

        print(f"Loading chunks from {self.input_file}")
        with open(self.input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunks.append(chunk)

        print(f"Loaded {len(chunks)} chunks")
        return chunks

    def process_and_store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Process chunks in batches and store them in ChromaDB."""
        # Extract data for ChromaDB
        chunk_ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{"source": chunk["source"]} for chunk in chunks]

        print(f"Processing {len(chunks)} chunks in batches of {self.batch_size}")

        # Process and add in batches
        for i in tqdm(range(0, len(chunks), self.batch_size)):
            batch_ids = chunk_ids[i : i + self.batch_size]
            batch_texts = texts[i : i + self.batch_size]
            batch_metadatas = metadatas[i : i + self.batch_size]

            # Embeddings are computed by our custom embedding function when using the add method
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,  # type: ignore
            )

        print(f"Successfully stored {len(chunks)} chunks in ChromaDB")

        # Get collection stats
        collection_count = self.collection.count()
        print(f"Total documents in collection: {collection_count}")

    def run(self) -> None:
        """Run the full vectorization process."""
        chunks = self.load_chunks()
        self.process_and_store_chunks(chunks)
        print(
            f"Vector database created successfully at: {os.path.abspath(self.db_directory)}"
        )
        print("You can now query the database using ChromaDB's query API.")


class OpenAIEmbeddingFunction:
    """Custom embedding function for ChromaDB using OpenAI API."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize with a specific model.

        Args:
            model_name: Name of the OpenAI embedding model to use
            max_retries: Maximum number of retries for API calls
            retry_delay: Initial delay between retries (in seconds)
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dimension = self._get_embedding_dimension()
        print(f"Using OpenAI API with model: {model_name}")
        print(f"Embedding dimension: {self.dimension}")

    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        # BGE-M3 generates embeddings with 1024 dimensions
        if "bge-m3" in self.model_name.lower():
            return 1024
        # Default to 1536 for text-embedding-ada-002
        return 1536

    def _call_api_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI API with retry logic."""
        retry_count = 0
        delay = self.retry_delay

        while retry_count < self.max_retries:
            try:
                response = client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    encoding_format="float",
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise Exception(
                        f"Failed to get embeddings after {self.max_retries} retries: {e}"
                    )

                print(
                    f"API call failed, retrying in {delay} seconds... (Attempt {retry_count}/{self.max_retries})"
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff

        # This line should never be reached because we either return from the try block
        # or raise an exception in the except block when retries are exhausted
        raise Exception("Unexpected execution path in _call_api_with_retry")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using OpenAI API.

        Args:
            input: List of texts to embed (parameter name must be 'input' for ChromaDB)

        Returns:
            List of embeddings as float lists
        """
        return self._call_api_with_retry(input)


if __name__ == "__main__":
    # Load environment variables from .env.local if it exists
    env_file = ".env.local"
    if os.path.exists(env_file):
        dotenv.load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")

    parser = argparse.ArgumentParser(
        description="Generate embeddings from text chunks using OpenAI API and store them in ChromaDB."
    )
    parser.add_argument("--input", "-i", help="Input JSONL file containing text chunks")
    parser.add_argument(
        "--db",
        "-d",
        default="artifacts/vector_stores/chroma_db",
        help="Directory where ChromaDB will store the vector database (default: artifacts/vector_stores/chroma_db)",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        help="OpenAI API key (if not provided, will try to use OPENAI_API_KEY from environment or .env.local file)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="BAAI/bge-m3",
        help="Name of the embedding model to use (default: BAAI/bge-m3)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )

    args = parser.parse_args()

    if not args.input:
        parser.error("Input file is required")

    vectorizer = ChunkVectorizerAPI(
        input_file=args.input,
        db_directory=args.db,
        api_key=args.api_key,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    vectorizer.run()
