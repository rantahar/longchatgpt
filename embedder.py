import pickle
import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


with open('api_key', 'r') as file1:
    openai.api_key = file1.readlines()[0].strip()

os.environ['OPENAI_API_KEY'] = openai.api_key


class Memory():
    def __init__(
        self,
        memory_file,
        messages
    ):
        self.memory_file = "memory/" + memory_file
        self.encode_conversation(messages)
        
        
    def split_message(self, message, max_split_len = 200):
        # split the message
        splits = message["content"].split("\n")
        splits = [s for s in splits if len(s) > 0]

        #combine small chunks, keeping below max_split_len
        combined_splits = []
        current_split = ""
        for split in splits:
            if len(split) > 2*max_split_len:
                print("long split", len(split))
            if len(current_split) + len(split) <= max_split_len:
                if current_split:
                    current_split += "\n" + split
                else:
                    current_split = split
            else:
                if current_split:
                    combined_splits.append(current_split)
                current_split = split
        if current_split:
            combined_splits.append(current_split)

        return combined_splits
    
    def encode_new_messages(self, messages, N=2):
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
                print(texts)
                self.db = FAISS.from_texts(texts, OpenAIEmbeddings())

                with open(self.memory_file, "wb") as file:
                    pickle.dump(self.db, file)

    def query(self, key, k=20):
        return self.db.similarity_search(key, k=k)

