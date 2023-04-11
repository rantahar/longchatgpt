import json
import openai
from tokens import num_tokens_from_messages
import os


with open('api_key', 'r') as file1:
    openai.api_key = file1.readlines()[0].strip()


class LongChat():
    def __init__(
        self,
        conversation = "summary_chatgpt.json",
        max_summary_length = 120,
        model = "gpt-3.5-turbo",
        summarize_every = 3,
        max_tokens = 1024,
        conversations_path = "conversations/"
    ):
        self.conversations_path = conversations_path
        self.conversation = conversation
        self.messages = []
        self.index = 0
        self.summarize_every = summarize_every
        self.model = model
        self.summary_prompt = "Please provide a short but complete summary of the conversation so far. The summary should include the main ideas, themes, and topics covered, as well as any key takeaways or conclusions reached. The summary should be no more than 50-60 words. The summary is for you (gpt-3.5-turbo) and does not need to be human-readable."
        self.max_summary_length = max_summary_length
        self.max_tokens = max_tokens

        self.read_conversation(conversation)

    def read_conversation(self, conversation):
        self.conversation = conversation
        self.messages_file = os.path.join(self.conversations_path, conversation)
        with open(self.messages_file, 'r') as f:
            self.messages = json.load(f)
    
        self.messages.insert(0, {"role": "system", "content": "You are a helpful assistant. You will occationally summarize the conversation for yourself when prompted. You will use existing summaries to continue the conversation naturally."})
    
        # Define initial variables
        self.index = len(self.messages)//2
    
    def shorten_conversation(self):
        short = self.messages[-20:]
        while num_tokens_from_messages(short) > self.max_tokens:
            short=short[1:]
        print(f"sending {len(short)} messages with {num_tokens_from_messages(short)} tokens")
        return short
    
    def summarize_chatgpt(self):
        print("summarizing")
        self.messages.append({"role": "user", "content": self.summary_prompt})
        result = openai.ChatCompletion.create(
          model=self.model, messages=self.shorten_conversation()
        )
        self.messages.append({"role": "assistant", "content": result.choices[0].message.content})

    def summarize_api(self):
        print("summarizing")
        self.messages.append({"role": "user", "content": self.summary_prompt})
        result = openai.Completion.create(
          engine="davinci",
          prompt=(f"{self.summary_prompt}:\n{self.shorten_conversation()}\n\nSummary:"),
          max_tokens=self.max_tokens,
          temperature=0.5,
          n = 1,
          stop=None,
          frequency_penalty=0,
          presence_penalty=0
        ).choices[0].text.strip()
        self.messages.append({"role": "assistant", "content": result})

    def new_message(self, user_message):
        print(self.index)
        self.messages.append({"role": "user", "content": user_message})
        try: 
            result = openai.ChatCompletion.create(
              model=self.model, messages=self.shorten_conversation()
            )
        except Exception as e:
            self.messages.pop()
            raise(e)
        
        self.messages.append({"role": "assistant", "content": result.choices[0].message.content})

        with open(self.messages_file, 'w') as outfile:
            json.dump(self.messages[1:], outfile, indent=4)
        
        if self.index % self.summarize_every == 0 and self.index > 0:
            try:
                self.summarize_chatgpt()    
            except Exception as e:
                self.messages.pop()
                raise(e)

            with open(self.messages_file, 'w') as outfile:
                json.dump(self.messages[1:], outfile, indent=4)

        self.index += 1



