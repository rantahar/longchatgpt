import json
from openai import OpenAI
from tokens import num_tokens_from_messages, count_tokens
import os
import functions
from Levenshtein import distance
import traceback
import embedder


with open('api_key', 'r') as file1:
    client = OpenAI(api_key=file1.readlines()[0].strip())


class LongChat():
    def __init__(
        self,
        conversation = "default.json",
        model = "gpt-4-1106-preview",
        #model = "gpt-4",
        #model = "gpt-3.5-turbo",
        #model = "gpt-3.5-turbo-16k",
        summarize_every = 5,
        summary_similarity_threshold = 0.2,
        content_tokens = 2000,
        memory_tokens = 400,
        summary_tokens = 200,
        reply_tokens = 1000,
        min_messages = 2,
        conversations_path = "conversations/"
    ):
        self.conversations_path = conversations_path
        self.conversation = conversation
        self.messages = []
        self.summary = {"role": "user", "content": "We have not started the conversation yet."}
        self.first_summary = True
        self.messages_since_summary = 0
        self.summarize_every = summarize_every
        self.summary_similarity_threshold = summary_similarity_threshold
        self.summary_rejected = False
        self.model = model
        self.system_message = \
"""You are a helpful AI assistant. You will use notes and the summary to maintain a coherent conversation and offer helpful advice.

- Your memory is limited, so store useful information in notes. Notes are stored permanently into files.
- When you generate code, you will be careful to not plagiarize any existing code.
"""
        self.summary_prompt = """Provide a short but complete summary of our current conversation, including topics covered, key takeaways and conclusions? This summary is for you (gpt-3.5-turbo), and does not need to be human readable. Make only small updates to the previous summary to maintain coherence and relevance. The summary must be less than 100 words."""
        self.summary_tokens = summary_tokens
        self.content_tokens = content_tokens
        self.memory_tokens = memory_tokens
        self.reply_tokens = reply_tokens
        self.min_messages = min_messages

        self.read_conversation(conversation)

    def read_conversation(self, conversation):
        print(traceback.format_exc())
        self.conversation = conversation
        self.messages_file = os.path.join(self.conversations_path, conversation)
        with open(self.messages_file, 'r') as f:
            content = json.load(f)

        self.messages = content.get("messages", [])
        self.summary = content.get("summary", {"role": "user", "content": "We have not started the conversation yet."})
        self.first_summary = content.get("first_summary", True)
        self.system_message = content.get("system_message", self.system_message)
        self.messages_since_summary = content.get("messages_since_summary", 0)
        self.disable_functions = content.get("disable_functions", False)
        self.memory_file = content.get("memory_file", self.conversation.replace(".json", ".pickle"))
        
        self.vector_memory = embedder.Memory(self.memory_file, self.messages)

    def build_notes_message(self, messages):
        self.vector_memory.encode_conversation(self.messages)

        if len(messages) == 0:
            return None
        
        key = messages[-1]["content"]
        result = self.vector_memory.query(key, k=20)
        
        notes_message = "Parts from previous conversation you remember:"
        for page in result:
            note = page.page_content
            if any(note in m["content"] for m in messages):
                continue
            if note in notes_message:
                continue
            notes_message += f"\n\n{note}"
            if count_tokens(notes_message) >= self.memory_tokens:
                break

        return notes_message
    
    def last_message_index(self, messages=None):
        if messages is None:
            messages = self.messages
        index = -min([len(messages), 200])
        while num_tokens_from_messages(messages[index:]) > self.content_tokens:
            index=index+1
        if index > -self.min_messages:
            index = -self.min_messages
        print(f"{-index} messages with {num_tokens_from_messages(messages[index:])} tokens (max {self.content_tokens})")
        return index

    def old_messages(self):
        return list(self.messages[:self.last_message_index()])

    def new_messages(self, messages=None):
        """ Get the latest messages that fit in the token limit """
        if messages is None:
            messages = self.messages
        index = self.last_message_index(messages)
        short = messages[index:]
        
        summary = f"This is a summary you wrote for yourself: {self.summary['content']}"
        notes = self.build_notes_message(short)
        if notes:
            short.insert(0, {"role": "system", "content": notes})
        short.insert(0, {"role": "system", "content": summary})
        short.insert(0, {"role": "system", "content": self.system_message})
        return short

    def add_message(self, role, message_content):
        self.messages.append({"role": role, "content": message_content})
        self.vector_memory.encode_new_messages(self.messages)
        self.dump_conversation()
    
    def messages_to_send(self, messages=None):
        """ Get messages and include notes in the system message """
        messages = self.new_messages(messages)
        return messages
    
    def summarize(self, messages=None):
        print("summarizing")
        messages = self.new_messages(messages)
        messages.append({"role": "user", "content": self.summary_prompt})
        summary = client.chat.completions.create(model=self.model, messages=messages).choices[0].message.content
        
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

    def dump_conversation(self):
        with open(self.messages_file, 'w') as outfile:
            json.dump({
                "summary": self.summary,
                "first_summary": self.first_summary,
                "messages": self.messages,
                "system_message": self.system_message,
                "messages_since_summary": self.messages_since_summary,
                "disable_functions": self.disable_functions
            }, outfile, indent=4)

    def check_summary(self):
        if (self.messages_since_summary >= self.summarize_every) or self.first_summary:
            self.summarize()
            if not self.summary_rejected:
                self.messages_since_summary = 0

            self.dump_conversation()

    def handle_function_call(self, function_call):
        print("function call message:", function_call)
        function_name = function_call["name"]
        parameters = function_call.arguments
        if function_name in functions.implementations:
            function = functions.implementations[function_name]
            try:
                result, system_note = function(**json.loads(parameters))
            except Exception as e:
                print(e)
                print("exception in function call")
                result = "exception in function call:" +str(e)
                system_note = None
        else:
            print("no such function")
            result = "no such function"
            system_note = None
        

        messages = self.new_messages()
        messages.append({"role": "function", "name": function_name, "content": result})
        messages = self.messages_to_send(messages)
        result = client.chat.completions.create(model=self.model,
        messages=messages)
        if system_note:
            self.messages.append({"role": "system", "content": system_note})
        
        message = result.choices[0].message.content
        self.add_message("assistant", message)

        self.messages_since_summary += 1
        self.dump_conversation()

        self.check_summary()
        return result
    
    def post_user_message(self, user_message):
        if user_message != "":
            self.add_message("user", user_message)
    
    def request_ai_message(self, disable_function_calls=False):
        print("messages_since_summary", self.messages_since_summary, self.first_summary)

        new_messages = self.messages_to_send()
        print(f"sending {len(new_messages)} messages with {num_tokens_from_messages(new_messages)} tokens")
        try:
            if disable_function_calls == "on":
                self.disable_functions = "on"
                result = client.chat.completions.create(model=self.model, messages=new_messages,
                max_tokens = self.reply_tokens)
            else:
                self.disable_functions = "off"
                result = client.chat.completions.create(model=self.model, messages=new_messages,
                max_tokens = self.reply_tokens,
                functions=functions.definitions,
                function_call="auto")
            print(result)
        except Exception as e:
            self.messages.pop()
            raise(e)
        

        message = result.choices[0].message
        while message.tool_calls:
            return message
        
        message = result.choices[0].message.content
        self.add_message("assistant", message)
        
        self.messages_since_summary += 1
        self.dump_conversation()

        self.check_summary()
        return {}
    

