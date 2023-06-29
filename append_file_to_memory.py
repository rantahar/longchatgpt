import sys
import os
import openai
import embedder
import PyPDF2
import tqdm

with open('api_key', 'r') as file1:
    openai.api_key = file1.readlines()[0].strip()

os.environ['OPENAI_API_KEY'] = openai.api_key


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

def encode_and_append(text, memory_file):
    vector_memory = embedder.Memory(memory_file, [])
    vector_memory.encode_text(text)

if __name__ == "__main__":
    if len(sys.argv) not in [3,4]:
        print("Usage: python append_file_to_memory.py <input_file> <memory_file> [first_chunk]")
    else:
        input_file = sys.argv[1]
        memory_file = sys.argv[2]
        if len(sys.argv) > 3:
            start_chunk = int(sys.argv[3])
        else:
            start_chunk = 0

        if input_file.endswith('.pdf'):
            text = extract_text_from_pdf(input_file)
            
        else:
            with open(input_file, 'r') as file:
                text = file.read()

        vector_memory = embedder.Memory(memory_file, [])

        texts = vector_memory.split_text(text)

        chunk_size = 100
        for i in tqdm.tqdm(range(start_chunk*chunk_size, len(texts), chunk_size), desc='Chunks'):
            vector_memory.encode_texts(texts[i:i+chunk_size])



