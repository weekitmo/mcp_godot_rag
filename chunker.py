#!/usr/bin/env python3

import os
import json
import hashlib
import argparse
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    """Process markdown files into chunks and save as JSONL."""
    
    def __init__(
        self, 
        input_dir: str, 
        chunk_size: int = 400,
        chunk_overlap: int = 20
    ):
        """Initialize with input/output paths and chunking parameters."""
        self.input_dir = input_dir
        
        # Get the base domain name from directory
        base_domain = os.path.basename(input_dir)
        # Ensure consistent domain naming even if passed through different pipeline stages
        if '.' in base_domain:
            base_domain = base_domain.replace('.', '-')
            
        self.output_file = f"artifacts/chunks/{base_domain}_chunks_SZ_{chunk_size}_O_{chunk_overlap}.jsonl"
        
        # Initialize LangChain's RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language="markdown",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        
    def _generate_uid(self, file_path: str) -> str:
        """Generate a unique ID based on file path."""
        # Create a hash of the file path
        return hashlib.md5(file_path.encode()).hexdigest()
        
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single markdown file into chunks."""
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Generate a unique ID based on file path
            uid = self._generate_uid(file_path)
            
            # Split content into chunks using LangChain's splitter
            chunks = self.splitter.split_text(content)
            
            # Create JSON objects for each chunk
            result = []
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue
                    
                chunk_data = {
                    'id': f'{uid}-{i}',
                    'text': chunk,
                    'source': file_path
                }
                result.append(chunk_data)
                
            print(f"Processed {file_path}: {len(result)} chunks")
            return result
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
            
    def process_directory(self) -> List[Dict[str, Any]]:
        """Process all markdown files in the input directory."""
        all_chunks = []
        file_count = 0
        
        # Walk through the input directory
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    chunks = self.process_file(file_path)
                    all_chunks.extend(chunks)
                    file_count += 1
                    
        print(f"\nProcessed {file_count} files with a total of {len(all_chunks)} chunks.")
        return all_chunks
        
    def save_jsonl(self, chunks: List[Dict[str, Any]]) -> None:
        """Save chunks to JSONL file."""
        # Create directory for output file if needed
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Delete the output file if it exists
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            
        # Write chunks to JSONL file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                
        print(f"Saved {len(chunks)} chunks to {self.output_file}")
        
    def run(self) -> None:
        """Run the full chunking process."""
        chunks = self.process_directory()
        self.save_jsonl(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split markdown files into chunks and save as JSONL.")
    parser.add_argument("--input", "-i", 
                        help="Input directory containing markdown files")
    parser.add_argument("--chunk-size", "-s", type=int, default=400,
                        help="Maximum size of chunks in characters (default: 400)")
    parser.add_argument("--chunk-overlap", "-v", type=int, default=20,
                        help="Overlap between chunks in characters (default: 20)")
    
    args = parser.parse_args()
    
    chunker = TextChunker(
        input_dir=args.input,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    chunker.run()