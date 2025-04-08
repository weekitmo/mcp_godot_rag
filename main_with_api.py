from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from mcp.server.fastmcp import FastMCP
import chromadb
import os
import argparse
import signal
import sys
import openai

mcp = FastMCP("Godot RAG Server")

collection: chromadb.Collection
base_url = "https://api.siliconflow.cn/v1"

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
parser.add_argument(
    "--api-key",
    "-k",
    help="OpenAI API key (if not provided, will try to use OPENAI_API_KEY from environment or .env.local file)",
)

args = parser.parse_args()

openai_client = openai.Client(base_url=base_url, api_key=args.api_key)


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
        embeddings = openai_client.embeddings.create(model="BAAI/bge-m3", input=[query])

        results = collection.query(
            query_embeddings=embeddings.data[0].embedding, n_results=20
        )

        # 添加检查确保结果不是None且包含预期的结构
        if results is None or "documents" not in results:
            return ["No documents found"]

        if not results["documents"] or len(results["documents"]) == 0:
            return ["No documents found"]

        if results["documents"][0] is None:
            return ["No document content found"]

        # 现在可以安全地访问文档内容
        documents = results["documents"][0][:]
        return documents
    except Exception as e:
        raise Exception(f"Failed to query ChromaDB: {str(e)}")


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down gracefully. Please wait...")
        # Close ChromaDB resources
        if "client" in globals():
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
    print("Starting Godot RAG MCP Server...")
    print(f"ChromaDB path: {args.chromadb_path}")
    print(f"Collection name: {args.collection_name}")
    try:
        client = chromadb.PersistentClient(path=args.chromadb_path)
        # 显式传递 embedding_function=None 来避免类型不兼容问题
        collection = client.get_collection(name=args.collection_name, embedding_function=None)
        print(f"Collection {args.collection_name} loaded successfully")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Shutting down...")
        # No need to do additional cleanup here as the signal handler will be called
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
