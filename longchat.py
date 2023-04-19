import json
import openai
from tokens import num_tokens_from_messages
import os
import numpy as np
from Levenshtein import distance

with open('api_key', 'r') as file1:
    openai.api_key = file1.readlines()[0].strip()


class LongChat():
    def __init__(
        self,
        conversation = "summary_chatgpt.json",
        max_summary_length = 120,
        model = "gpt-3.5-turbo",
        summarize_every = 2,
        summary_similarity_threshold = 0.2,
        max_tokens = 512,
        min_messages = 3,
        conversations_path = "conversations/"
    ):
        self.conversations_path = conversations_path
        self.conversation = conversation
        self.messages = []
        self.summary = {"role": "user", "content": "We have not started the conversation yet."}
        self.first_summary = True
        self.index = 0
        self.summarize_every = summarize_every
        self.summary_similarity_threshold = summary_similarity_threshold
        self.summary_rejected = False
        self.model = model
        self.system_message = """You are a helpful assistant. You will use conversation summaries to rember the context of the conversation and to continue it naturally."""
        self.summary_prompt = """Provide a short but complete summary of our current conversation, including topics covered, key takeaways and conclusions? This summary is for you (gpt-3.5-turbo), and does not need to be human readable. Make only small updates to the previous summary to maintain coherence and relevance."""
        self.max_summary_length = max_summary_length
        self.max_tokens = max_tokens
        self.min_messages = min_messages

        self.read_conversation(conversation)

    def read_conversation(self, conversation):
        self.conversation = conversation
        self.messages_file = os.path.join(self.conversations_path, conversation)
        with open(self.messages_file, 'r') as f:
            content = json.load(f)
        if "messages" in content.keys():
            self.messages = content["messages"]
        if "summary" in content.keys():
            self.summary = content["summary"]
            if self.summary["content"] != "We have not started the conversation yet.":
                self.first_summary = False

        # Define initial variables
        self.index = len(self.messages)//2
    
    def last_message_index(self, messages=None):
        if messages is None:
            messages = self.messages
        index = -min([len(messages), 20])
        while num_tokens_from_messages(messages[index:]) > self.max_tokens:
            index=index+1
        if index > -self.min_messages:
            index = -self.min_messages
        print(f"{-index} messages with {num_tokens_from_messages(messages[index:])} tokens")
        return index

    def old_messages(self):
        return list(self.messages[:self.last_message_index()])

    def new_messages(self, messages=None):
        if messages is None:
            messages = self.messages
        index = self.last_message_index(messages)
        short = messages[index:]
        short.insert(0, {"role": "system", "content": f"This is a summary you wrote for yourself: {self.summary['content']}"})
        short.insert(0, {"role": "system", "content": self.system_message})
        return short
    
    def summarize_chatgpt(self, messages=None):
        print("summarizing")
        messages = self.new_messages(messages)
        messages.append({"role": "user", "content": self.summary_prompt})
        summary = openai.ChatCompletion.create(
          model=self.model, messages=messages
        ).choices[0].message.content
        
        if self.first_summary:
            self.summary = {"role": "assistant", "content": summary}
            self.first_summary = False
            return

        old_summary = self.summary["content"]
        similarity = 1 - distance(old_summary.lower(), summary.lower()) / max(len(old_summary), len(summary))
        if similarity >= self.summary_similarity_threshold:
            self.summary = {"role": "assistant", "content": summary}
            self.summary_rejected= False
        else:
            print("Divergent summary rejected", similarity)
            print(old_summary)
            print(summary)
            self.summary_rejected = True


    def summarize_api(self, messages):
        print("summarizing")
        messages = self.new_messages()
        self.messages.append({"role": "user", "content": self.summary_prompt})
        print(f"sending {len(messages)} messages with {num_tokens_from_messages(messages)} tokens")
        result = openai.Completion.create(
          engine="davinci",
          prompt=(f"{self.summary_prompt}:\n{messages}\n\nSummary:"),
          max_tokens=self.max_tokens,
          temperature=0.5,
          n = 1,
          stop=None,
          frequency_penalty=0,
          presence_penalty=0
        ).choices[0].text.strip()
        self.messages.append({"role": "assistant", "content": result})

    def dump_conversation(self):
        with open(self.messages_file, 'w') as outfile:
            json.dump({"summary": self.summary, "messages": self.messages}, outfile, indent=4)

    def new_message(self, user_message):
        print(self.index)
        if user_message != "":
            self.messages.append({"role": "user", "content": user_message})
        new_messages = self.new_messages()
        print(f"sending {len(new_messages)} messages with {num_tokens_from_messages(new_messages)} tokens")
        try: 
            result = openai.ChatCompletion.create(
              model=self.model, messages=new_messages
            )
        except Exception as e:
            self.messages.pop()
            raise(e)
        
        self.messages.append({"role": "assistant", "content": result.choices[0].message.content})
        
        self.dump_conversation()

        if self.summary_rejected or self.first_summary or self.index % self.summarize_every == 0 and self.index > 0:
            try:
                self.summarize_chatgpt()
            except Exception as e:
                self.messages.pop()
                raise(e)

            self.dump_conversation()

        self.index += 1
    

