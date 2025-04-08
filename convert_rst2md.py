#!/usr/bin/env python3
"""
RST to Markdown converter for Godot documentation.
This script converts RST files from specified directories to Markdown format
and preserves the directory structure in the output directory.
"""

import os
import sys
import re
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import shutil

# You might need to install these packages
try:
    from docutils.core import publish_parts
    from docutils.writers.html4css1 import Writer
    import pypandoc
except ImportError:
    print("Required dependencies not found. Installing...")
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "docutils", "pypandoc"]
    )
    from docutils.core import publish_parts
    from docutils.writers.html4css1 import Writer
    import pypandoc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def rst_to_markdown(rst_content, src_dir=None):
    """
    Convert RST content to Markdown using a two-step process:
    1. RST to HTML using docutils
    2. HTML to Markdown using pypandoc
    """
    try:
        # Convert RST to HTML
        parts = publish_parts(
            rst_content, writer=Writer(), settings_overrides={"report_level": 5}
        )
        
        # Check if html_body exists in the returned dictionary
        if "html_body" not in parts:
            logger.error("html_body not found in the published parts")
            return None
            
        html = parts["html_body"]

        # Convert HTML to Markdown
        markdown = pypandoc.convert_text(html, "md", format="html")

        # Clean up the markdown
        markdown = clean_markdown(markdown, src_dir)

        return markdown
    except Exception as e:
        logger.error(f"Error converting RST to markdown: {e}")
        return None


def clean_markdown(markdown, src_dir=None):
    """Clean up and format the markdown content"""
    # Fix broken links
    markdown = re.sub(r"\[([^\]]+)\]\{(.+?)\}", r"[\1](\2)", markdown)

    # Fix image links (ensure they point to original location)
    def fix_image_link(match):
        alt_text = match.group(1)
        img_path = match.group(2)

        # Don't migrate image files, just update references to them
        if "../" in img_path or img_path.startswith('/'):
            # Handle relative or absolute paths
            return f"![{alt_text}]({img_path})"
        else:
            # Handle paths relative to the original docs directory
            if src_dir is None:
                return f"![{alt_text}]({img_path})"
            return f"![{alt_text}](../{os.path.basename(src_dir)}/{img_path})"

    markdown = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fix_image_link, markdown)

    # Additional cleanup as needed
    markdown = markdown.replace("\\_", "_")  # Fix escaped underscores

    return markdown


def process_file(src_file, dst_file, base_src_dir, base_dst_dir):
    """Process a single RST file and convert it to markdown"""
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        # Read RST content
        with open(src_file, "r", encoding="utf-8") as f:
            rst_content = f.read()

        # Convert RST to Markdown
        markdown_content = rst_to_markdown(rst_content, base_src_dir)

        if markdown_content:
            # Write Markdown content
            with open(dst_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            rel_path = os.path.relpath(src_file, base_src_dir)
            logger.info(f"Converted: {rel_path}")
            return True
        else:
            logger.error(f"Failed to convert: {src_file}")
            return False
    except Exception as e:
        logger.error(f"Error processing file {src_file}: {e}")
        return False


# Define a helper function outside to avoid pickling issues with lambda
def process_file_helper(args):
    return process_file(*args)


def process_directory(src_dir, dst_dir, exclude_dirs=None):
    """
    Process all RST files in the source directory and convert them to markdown files
    in the destination directory, preserving the directory structure.
    """
    if exclude_dirs is None:
        exclude_dirs = ["img", "video"]

    base_src_dir = src_dir
    base_dst_dir = dst_dir

    # Get the list of all RST files
    rst_files = []
    for root, dirs, files in os.walk(src_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".rst"):
                src_file = os.path.join(root, file)

                # Compute the destination path, preserving directory structure
                rel_path = os.path.relpath(src_file, base_src_dir)
                dst_file = os.path.join(base_dst_dir, rel_path.replace(".rst", ".md"))

                rst_files.append((src_file, dst_file, base_src_dir, base_dst_dir))

    # Process files with a process pool to handle memory efficiently
    total_files = len(rst_files)
    logger.info(f"Found {total_files} RST files to convert")

    successful = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_file_helper, rst_files))
        successful = sum(1 for result in results if result)

    logger.info(f"Successfully converted {successful} out of {total_files} files")
    return successful


def main():
    parser = argparse.ArgumentParser(
        description="Convert RST files to Markdown format."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="docs",
        help="Source directory containing RST files",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="artifacts",
        help="Destination directory for converted Markdown files",
    )
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = args.dst

    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Process index.rst separately
    index_src = os.path.join(src_dir, "index.rst")
    index_dst = os.path.join(dst_dir, "index.md")

    if os.path.exists(index_src):
        logger.info("Processing index.rst")
        process_file(index_src, index_dst, src_dir, dst_dir)

    # Process the specified directories
    target_dirs = ["classes", "tutorials", "getting_started"]

    for dir_name in target_dirs:
        dir_src = os.path.join(src_dir, dir_name)
        dir_dst = os.path.join(dst_dir, dir_name)

        if os.path.isdir(dir_src):
            logger.info(f"Processing directory: {dir_name}")
            process_directory(dir_src, dir_dst)
        else:
            logger.warning(f"Directory not found: {dir_name}")

    logger.info("Conversion completed")


if __name__ == "__main__":
    main()
