from mcp.server.fastmcp import FastMCP
import chromadb
from sentence_transformers import SentenceTransformer
import argparse
import signal
import sys
import asyncio

mcp = FastMCP("Godot RAG Server")

client: chromadb.PersistentClient
collection: chromadb.Collection
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@mcp.tool()
def get_godot_context(query: str) -> list:
    """
    Godot engine has evolved a lot, and a lot of the pretrained knowledge is outdated and cannot be relied on.
    This tool retrieves a list of the latest relevant Godot documentation snippets based on the provided query.
    If user askes anything related to the Godot engine, including api and class references, even you are confident,
    this function should still be called. If there is any conflict between your knowledge and the retrieved snippets,
    the snippets should be considered more reliable, otherwise it's okay to rely on your knowledge. Only call this
    function if you are certain it's about the Godot engine.

    Args:
        query: keywords related to Godot engine

    Returns:
        list of relevant Godot documentation/references snippets
    """
    try:
        results = collection.query(
            query_embeddings=model.encode([query]).astype(float).tolist(), n_results=20
        )

        # based on your data, you may include other info such as metadata, etc.
        documents = results["documents"][0][:]

        return documents
    except Exception as e:
        return {"error": f"Failed to query ChromaDB: {str(e)}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Godot RAG MCP Server")
    parser.add_argument(
        "--chromadb-path",
        "-d",
        type=str,
        required=True,
        help="Path to the ChromaDB database",
    )
    parser.add_argument(
        "--collection-name",
        "-c",
        type=str,
        required=True,
        help="Name of the ChromaDB collection to query",
    )

    args = parser.parse_args()

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down gracefully. Please wait...")
        # Close ChromaDB resources
        if 'client' in globals():
            try:
                # ChromaDB's PersistentClient doesn't have a _client attribute
                # Just let it be garbage collected
                print("ChromaDB client resources released.")
            except Exception as e:
                print(f"Error closing ChromaDB client: {e}")
        # Exit cleanly
        sys.exit(0)

    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        client = chromadb.PersistentClient(path=args.chromadb_path)
        collection = client.get_collection(args.collection_name)
        print(f"Collection {args.collection_name} loaded successfully")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Shutting down...")
        # No need to do additional cleanup here as the signal handler will be called
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
