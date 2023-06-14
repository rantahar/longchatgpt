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
        conversation = "default.json",
        max_summary_length = 120,
        model = "gpt-3.5-turbo-0613",
        summarize_every = 3,
        summary_similarity_threshold = 0.2,
        max_tokens = 3000,
        min_messages = 6,
        conversations_path = "conversations/"
    ):
        self.conversations_path = conversations_path
        self.conversation = conversation
        self.messages = []
        self.summary = {"role": "user", "content": "We have not started the conversation yet."}
        self.notes =  {}
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
        self.summary_prompt = """Provide a short but complete summary of our current conversation, including topics covered, key takeaways and conclusions? This summary is for you (gpt-3.5-turbo), and does not need to be human readable. Make only small updates to the previous summary to maintain coherence and relevance."""
        self.note_prompt = """Extract short, useful notes from the summary above. Provide at most 3 notes. The notes should help you (chatGPT) remember the conversation, so don't store information chatGPT already knows.

Each note must follow the format:
keyword list : note message : note importance (0-1)

For example:
chatbot, assistant, AI: You are a helpful AI assistant. : 0.1

Your reply must only contain notes following this syntax, and no other text.
"""
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
        if "first_summary" in content.keys():
            self.first_summary = content["first_summary"]
        if "notes" in content.keys():
            self.notes = content["notes"]
        if "system_message" in content:
            self.system_message = content["system_message"]
        if "messages_since_summary" in content:
            self.messages_since_summary = content["messages_since_summary"]
        else:
            self.messages_since_summary = 0

    def extract_notes(self, message):
        for line in message.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                print(line)
                keyword_list, note_msg, note_imp = line.split(":")
            except ValueError:
                continue
            # Create a list of individual keywords
            # If a keyword already exists in the dict, append the note message and importance to that keyword's tuple
            self.notes.append({
                "keywords": keyword_list.strip().lower(),
                "note": note_msg.strip().lower(),
                "importance": note_imp.strip().lower()
            })

    def build_notes_message(self, messages, max_notes = 5):
        messages = "\n".join([message["content"] for message in messages]).lower()
        
        matching_notes = []
        for note in self.notes:
            loc = -1
            keywords = note["keywords"].split(", ")
            for keyword in keywords:
                k_loc = messages.rfind(keyword)
                if k_loc > loc:
                    loc = k_loc
            if loc != -1:
                matching_notes.append(note.copy())
                matching_notes[-1]["loc"] = loc

        if len(matching_notes) == 0:
            return ""

        # Sort by importance
        matching_notes = sorted(matching_notes, key=lambda x:x["loc"], reverse=True)
        
        notes_message = "Useful notes:"
        for note in matching_notes[-max_notes:]:
            notes_message += f"\n{note['keywords']} - {note['note']}"
            print(note["loc"])

        return notes_message
    
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

        summary = f"This is a summary you wrote for yourself: {self.summary['content']}"
        notes = self.build_notes_message(short)
        if notes:
            summary += "\n\n" + notes
        short.insert(0, {"role": "system", "content": summary})
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

    def create_notes(self, messages=None):
        print("creating notes")
        summary = f"This is a summary you wrote for yourself: {self.summary['content']}"
        messages = [{"role": "assistant", "content": summary}]
        messages.append({"role": "user", "content": self.note_prompt})
        message = openai.ChatCompletion.create(
          model=self.model, messages=messages
        ).choices[0].message.content
        print(message)
        self.extract_notes(message)

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
            json.dump({
                "summary": self.summary,
                "first_summary": self.first_summary,
                "messages": self.messages,
                "notes": self.notes,
                "system_message": self.system_message,
                "messages_since_summary": self.messages_since_summary
            }, outfile, indent=4)

    def new_message(self, user_message):
        print("messages_since_summary", self.messages_since_summary, self.first_summary)
        if user_message != "":
            self.messages.append({"role": "user", "content": user_message})
        new_messages = self.new_messages()
        print(f"sending {len(new_messages)} messages with {num_tokens_from_messages(new_messages)} tokens")
        try: 
            result = openai.ChatCompletion.create(
              model=self.model, messages=new_messages,
              functions=[
                {
                    "name": "update_summary",
                    "description": "Update the current conversation summary. Use this when new useful information is available.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "A summary of the current conversation.",
                            }
                        },
                        "required": ["location"],
                    },
                }
            ],
            function_call="auto",
            )
        except Exception as e:
            self.messages.pop()
            raise(e)
        
        message = result.choices[0].message

        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            summary = message.get("summary")
            print(message)
            result = openai.ChatCompletion.create(
                model=self.model,
                messages=new_messages + [
                    message,
                    {
                        "role": "function",
                        "name": function_name,
                        "content": "summary updated",
                    },
                ],
            )

        
        message = result.choices[0].message.content
        self.messages.append({"role": "assistant", "content": message})
        
        self.messages_since_summary += 1
        self.dump_conversation()

        if (self.messages_since_summary >= self.summarize_every) or self.first_summary:
            self.summarize_chatgpt()
            self.create_notes()
            if not self.summary_rejected:
                self.messages_since_summary = 0

            self.dump_conversation()
    

