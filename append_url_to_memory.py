import sys
import requests
from bs4 import BeautifulSoup
import embedder


def append_webpage_content(url, memory_file):
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ').strip()
        text = '\n\n'.join([t.strip() for t in text.split("\n") if t.strip()])
        embedder.encode_and_append(text, memory_file)

def get_links(url, memory_file):
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        for link in soup.findAll('a'):
            if link.get('href') is not None:
                if link.get('href').endswith('.pdf'):
                    continue
                print(link.get('href'))
                append_webpage_content(link.get('href'), memory_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python append_webpage_to_memory.py [MEMORY_FILE] [URL]")
        sys.exit(1)

    url = sys.argv[2]
    memory_file = sys.argv[1]

    get_links(url, memory_file)

