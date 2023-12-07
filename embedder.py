import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from tokens import count_tokens


class Memory():
    def __init__(
        self,
        memory_file,
        messages
    ):
        self.memory_file = "memory/" + memory_file
        self.encode_conversation(messages)
    
        
    def split_text(self, text, max_split_len = 100, separator="\n"):
        # split the text
        splits = text.split(separator)
        splits = [s for s in splits if len(s) > 0]

        #combine small chunks, keeping below max_split_len
        combined_splits = []
        current_split = ""
        for split in splits:
            if count_tokens(split) > max_split_len:
                if separator == "\n":
                    new_sep = "."
                elif separator == ".":
                    new_sep = ","
                elif separator == ",":
                    new_sep = " "
                else:
                    return []
                new_splits = self.split_text(split, separator=new_sep, max_split_len=max_split_len)
                combined_splits += new_splits
                continue
            if count_tokens(current_split) + count_tokens(split) <= max_split_len:
                if current_split:
                    current_split += separator + split
                else:
                    current_split = split
            else:
                if current_split:
                    combined_splits.append(current_split)
                current_split = split
        if current_split:
            combined_splits.append(current_split)

        return combined_splits
    
    def split_message(self, message, max_split_len = 100, separator="\n\n"):
        return self.split_text(message["content"], max_split_len, separator)
    
    def encode_new_messages(self, messages, N=1):
        try:
            # If the pickled database is found, 
            with open(self.memory_file, "rb") as file:
                self.db = pickle.load(file)

            texts = []
            for message in messages[-N:]:
                texts += self.split_message(message)
            self.db.add_texts(texts)
        except:
            self.encode_conversation(messages)

        with open(self.memory_file, "wb") as file:
            pickle.dump(self.db, file)

    def encode_conversation(self, messages):
        try:
            with open(self.memory_file, "rb") as file:
                self.db = pickle.load(file)
            print("LOADED")
        except:
            if messages:
                texts = []
                for message in messages:
                    texts += self.split_message(message)
                hf = HuggingFaceEmbeddings()
                self.db = FAISS.from_texts(texts, hf)

                with open(self.memory_file, "wb") as file:
                    pickle.dump(self.db, file)

    def encode_texts(self, texts):
        with open(self.memory_file, "rb") as file:
            self.db = pickle.load(file)

        self.db.add_texts(texts)

        with open(self.memory_file, "wb") as file:
            pickle.dump(self.db, file)

    def encode_text(self, text):
        with open(self.memory_file, "rb") as file:
            self.db = pickle.load(file)

        texts = self.split_message({"content": text})
        self.db.add_texts(texts)

        with open(self.memory_file, "wb") as file:
            pickle.dump(self.db, file)

    def encode_text_file(self, text_file):
        with open(text_file, "r") as file:
            text = file.read()
        self.encode_text(text)

    def query(self, key, k=20):
        return self.db.similarity_search(key, k=k)



