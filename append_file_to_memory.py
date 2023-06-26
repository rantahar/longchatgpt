import sys
import embedder


def append_file_content(file_path, memory_file):
    with open(file_path, "r") as file:
        text = file.read()
    embedder.encode_and_append(text, memory_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python append_to_memory_script.py <input_file> <memory_file>")
    else:
        input_file = sys.argv[1]
        memory_file = sys.argv[2]
        append_file_content(input_file, memory_file)

