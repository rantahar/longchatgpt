import sys
import os
import openai
import embedder

with open('api_key', 'r') as file1:
    openai.api_key = file1.readlines()[0].strip()

os.environ['OPENAI_API_KEY'] = openai.api_key


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python append_file_to_memory.py <input_file> <memory_file>")
    else:
        input_file = sys.argv[1]
        memory_file = sys.argv[2]
        vector_memory = embedder.Memory(memory_file, [])
        vector_memory.encode_text_file(input_file)

