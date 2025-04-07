#!/usr/bin/env python3

import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime

# "dunzhang/stella_en_400M_v5"


class ChunkVectorizer:
    """Generate embeddings from text chunks and store them in a ChromaDB vector database."""

    def __init__(
        self, input_file: str, db_directory: str, model_name: str, batch_size: int = 32
    ):
        """Initialize the vectorizer with input path and model parameters.

        Args:
            input_file: Path to the input JSONL file containing text chunks
            db_directory: Directory where ChromaDB will store the vector database
            model_name: The name of the sentence-transformer model to use
            batch_size: Batch size for embedding generation
        """
        self.input_file = input_file
        self.db_directory = db_directory
        collection_base_name = os.path.basename(input_file).replace(".jsonl", "")

        # Create collection name and limit to 63 characters (database limitation)
        # Extract model name short version (e.g., "all-MiniLM-L6-v2" from full path)
        model_short_name = (
            model_name.split("/")[-1] if "/" in model_name else model_name
        )
        collection_name = f"{collection_base_name}_{model_short_name}"

        # Truncate if longer than 63 chars
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

        # Create embedding function using the sentence transformer
        self.embedding_function = CustomEmbeddingFunction(model_name)

        # Create a new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
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
            f.write(
                f"{self.collection_name} ({timestamp}) - Original model: {model_name}\n"
            )
        self.batch_size = batch_size

        # Load the embedding model
        print(f"Using model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(
            f"Model loaded: {model_name} (embedding dimension: {self.model.get_sentence_embedding_dimension()})"
        )

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

            # Embeddings are computed automatically by ChromaDB when using the add method
            self.collection.add(
                ids=batch_ids, documents=batch_texts, metadatas=batch_metadatas
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


class CustomEmbeddingFunction:
    """Custom embedding function for ChromaDB using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a specific model."""
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            input: List of texts to embed (parameter name must be 'input' for ChromaDB)

        Returns:
            List of embeddings as float lists
        """
        embeddings = self.model.encode(input)
        return embeddings.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings from text chunks and store them in ChromaDB."
    )
    parser.add_argument("--input", "-i", help="Input JSONL file containing text chunks")
    parser.add_argument(
        "--db",
        "-d",
        default="artifacts/vector_stores/chroma_db",
        help="Directory where ChromaDB will store the vector database (default: artifacts/vector_stores/chroma_db)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the sentence-transformer model to use (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )

    args = parser.parse_args()

    vectorizer = ChunkVectorizer(
        input_file=args.input,
        db_directory=args.db,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    vectorizer.run()
