import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from tokens import count_tokens


class Memory():
    def __init__(
        self,
        memory_file
    ):
        self.memory_file = "memory/" + memory_file
    
        
    def split_text(self, text, max_split_len = 100, separator="\n"):
        # split the text
        splits = text.split(separator)
        splits = [s for s in splits if len(s) > 0]

        #combine small chunks, keeping below max_split_len
        combined_splits = []
        current_split = ""
        for split in splits:
            if count_tokens(split) > max_split_len:
                if separator == "\n\n":
                    new_sep = "\n"
                elif separator == "\n":
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
            if (separator != "\n\n") and (count_tokens(current_split) + count_tokens(split) <= max_split_len):
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
            # If the pickled database is found
            with open(self.memory_file, "rb") as file:
                db = pickle.load(file)

            texts = []
            for message in messages[-N:]:
                splits = self.split_message(message)
                texts += [f"{message['role']}: {s}" for s in splits]
            db.add_texts(texts)
            with open(self.memory_file, "wb") as file:
                pickle.dump(db, file)
        except:
            self.encode_conversation(messages)

    def memory_exists(self):
        try:
            with open(self.memory_file, "rb") as file:
                db = pickle.load(file)
            return True
        except Exception as e:
            print(e)
            return False

    def encode_conversation(self, messages):
        if messages:
            texts = ["system: "]
            for message in messages:
                splits = self.split_message(message)
                texts += [f"{message['role']}: {s}" for s in splits]
            hf = HuggingFaceEmbeddings()
            db = FAISS.from_texts(texts, hf)

            with open(self.memory_file, "wb") as file:
                pickle.dump(db, file)

    def encode_texts(self, texts, source=None):
        if source:
            texts = [f"{source}: {t}" for t in texts if t]

        try:
            with open(self.memory_file, "rb") as file:
                db = pickle.load(file)
            db.add_texts(texts)
        except:
            db = FAISS.from_texts(texts, HuggingFaceEmbeddings())

        with open(self.memory_file, "wb") as file:
            pickle.dump(db, file)

    def encode_text(self, text, source=None):
        if len(text) == 0:
            return
        texts = self.split_text(text, separator="\n")
        if source:
            texts =  [f"{source}: {s}" for s in texts if s]

        try:
            with open(self.memory_file, "rb") as file:
                db = pickle.load(file)
            db.add_texts(texts)
        except:
            db = FAISS.from_texts(texts, HuggingFaceEmbeddings())

        with open(self.memory_file, "wb") as file:
            pickle.dump(db, file)

    def encode_text_file(self, text_file):
        with open(text_file, "r") as file:
            text = file.read()

        self.encode_text(text)

    def query(self, key, k=20):
        with open(self.memory_file, "rb") as file:
            db = pickle.load(file)
            
        return db.similarity_search(key, k=k)



