import os
import sys
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base URL to prepend to relative links
BASE_URL = os.environ.get(
    "BASE_URL", "https://github.com/pinecone-io/pinecone-python-client/blob/main/"
)


def replace_relative_links(html):
    soup = BeautifulSoup(html, "html.parser")

    # Find all anchor tags with an href attribute
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Skip if the link is already absolute or an anchor link
        if href.startswith(("http://", "https://", "#")):
            continue

        # Skip if the link is not a markdown file
        if not href.endswith(".md"):
            continue

        # Replace the relative link with an absolute URL
        new_href = urljoin(BASE_URL, href)
        print(f"{href} => {new_href}")
        a["href"] = new_href
    return str(soup)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix-relative-links.py input-dir [output-dir]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Recursively process all html files in the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".html"):
                continue

            print(f"Processing {file}")
            input_path = os.path.join(root, file)

            with open(input_path, "r", encoding="utf-8") as f:
                html = f.read()

            updated_html = replace_relative_links(html)

            if output_dir:
                # Get the relative path from input_dir to maintain folder structure
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                # Create the necessary subdirectories
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(updated_html)
            else:
                print(updated_html)
