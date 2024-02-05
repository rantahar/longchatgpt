import sys
import embedder


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python append_file_to_memory.py <query> <memory_file>")
    else:
        query = sys.argv[1]
        memory_file = sys.argv[2]
        vector_memory = embedder.Memory(memory_file, [])

        for d in vector_memory.query(query, k=20):
            print(d.page_content, "\n")

