import json
from openai import OpenAI
from anthropic import Anthropic
from tokens import num_tokens_from_messages, count_tokens
import os
import functions
from Levenshtein import distance
import traceback
import embedder
import copy


client = "anthropic"

    
class AnthropicClient():
    def __init__(
            self,
            api_key_file = "anthropic_key",
            #model = "claude-3-opus-20240229"
            model = "claude-3-sonnet-20240229",
            summarization_model="claude-3-haiku-20240307",
            input_tokens = 10000,
            output_tokens = 2000,
        ):
        with open(api_key_file, 'r') as file1:
            api_key = file1.readlines()[0].strip()

        self.model = model
        self.summarization_model = summarization_model
        self.api_key = api_key
        self.client = Anthropic(api_key=api_key)

        self.content_tokens = int(0.8*input_tokens)
        self.memory_tokens =  int(0.1*input_tokens)
        self.summary_tokens = int(0.05*input_tokens)
        self.reply_tokens =  output_tokens

    def coalesce_messages(self, messages):
        """ Anthropic does not like multiple messages from the same agent in a row. This function coalesces them. """
        coalesced_messages = []
        for message in messages:
            if len(coalesced_messages) == 0:
                coalesced_messages.append(message)
            elif coalesced_messages[-1]["role"] == message["role"]:
                coalesced_messages[-1]["content"] += "\n\n" + message["content"]
            else:
                coalesced_messages.append(message)
        return coalesced_messages
    
    def summarize_conversation(self, messages, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.summary_tokens
        conversation_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        prompt = f"Summarize the key points and main ideas from the following conversation snippet in a concise bullet point format:\n\n{conversation_text}"

        result = self.client.messages.create(
            model=self.summarization_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return result.content[0].text

    def request_message(self, messages, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.reply_tokens
        non_system_messages = [m for m in messages if m["role"] != "system"]
        system_message = "\n\n".join([m["content"] for m in messages if m["role"] == "system"])
        non_system_messages = self.coalesce_messages(non_system_messages)
        result = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_message,
            messages=non_system_messages
        )
        return result.content[0].text


class OpenAIClient():
    def __init__(
            self,
            api_key_file = "openai_key",
            model = "gpt-4o",
            summarization_model="gpt-3.5-turbo-0125",
            input_tokens = 5000,
            output_tokens = 2000,
        ):
        with open(api_key_file, 'r') as file1:
            api_key = file1.readlines()[0].strip()
        
        self.model = model
        self.summarization_model = summarization_model
        self.api_key = api_key
        
        self.content_tokens = int(0.8*input_tokens)
        self.memory_tokens =  int(0.1*input_tokens)
        self.summary_tokens = int(0.05*input_tokens)
        self.reply_tokens =  output_tokens

        self.client = OpenAI(api_key=api_key)

    def summarize_conversation(self, messages, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.summary_tokens
        conversation_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        prompt = f"Summarize the key points and main ideas from the following conversation snippet in a concise bullet point format:\n\n{conversation_text}"
        result = self.client.completions.create(
            model=self.summarization_model,
            max_tokens=max_tokens,
            prompt=prompt
        )
        return result.completion.text

    def request_message(self, messages, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.reply_tokens
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens
        ).choices[0].message.content


if client == "openai":
    client = OpenAIClient("openai_key")
elif client == "anthropic":
    client = AnthropicClient("anthropic_key")



class ConversationManager:
    def __init__(self, conversation="default.json", conversations_path="conversations/"):
        self.conversations_path = conversations_path
        self.conversation = conversation
        self.read_conversation()

    def read_conversation(self):
        self.messages_file = os.path.join(self.conversations_path, self.conversation)
        with open(self.messages_file, 'r') as f:
            content = json.load(f)

        if "system_message" not in content:
            raise ValueError("The 'system_message' key was not found in the conversation JSON file.")
        if "summary" not in content:
            raise ValueError("The 'summary' key was not found in the conversation JSON file.")

        self.messages = content.get("messages", [])
        self.summary = content.get("summary")
        self.system_message = content.get("system_message")
        self.first_summary = content.get("first_summary", True)
        self.messages_since_summary = content.get("messages_since_summary", 0)

        self.memory_file = content.get("memory_file", self.conversation.replace(".json", ".pickle"))

    def add_message(self, message=None, role=None, content=None):
        if message is None:
            if role is not None and content is not None:
                message = {"role": role, "content": content}
            else:
                raise ValueError("Either provide a message dictionary or both role and content")
        self.messages.append({"role": role, "content": content})
        self.save_conversation()

    def save_conversation(self):
        with open(self.messages_file, 'w') as outfile:
            json.dump({
                "summary": self.summary,
                "first_summary": self.first_summary,
                "messages": self.messages,
                "system_message": self.system_message,
                "messages_since_summary": self.messages_since_summary,
            }, outfile, indent=4)



class Summarizer:
    def __init__(
            self, model, summary_tokens, summarize_every, summary_similarity_threshold,
            conversation,
            summary_prompt = None
        ):
        self.model = model
        if summary_prompt is None:
            self.summary_prompt = """Provide a short but complete summary of our current conversation, including topics covered, key takeaways and conclusions? This summary is for you (gpt-3.5-turbo), and does not need to be human readable. Make only small updates to the previous summary to maintain coherence and relevance. The summary must be less than 200 words."""
        self.conversation = conversation
        self.summary_tokens = summary_tokens
        self.summarize_every = summarize_every
        self.summary_similarity_threshold = summary_similarity_threshold
        self.summary_rejected = False

    def set_conversation(self, conversation):
        self.conversation = conversation

    def summarize(self, messages):
        messages.append({"role": "user", "content": self.summary_prompt})
        summary = client.request_message(messages=messages)
        if self.conversation.first_summary:
            self.conversation.first_summary = False
            return summary
        else:
            return self.compare_and_update_summary(self.conversation.summary["content"], summary)
    
    def compare_and_update_summary(self, old_summary, new_summary):
        similarity = 1 - distance(old_summary.lower(), new_summary.lower()) / max(len(old_summary), len(new_summary))
        if similarity >= self.summary_similarity_threshold:
            self.summary_rejected = False
            return new_summary
        else:
            self.summary_rejected = True
            return old_summary

    def check_summary(self, context):
        self.conversation.messages_since_summary += 1
        if (self.conversation.messages_since_summary >= self.summarize_every) or self.conversation.first_summary:
            messages = context.in_context_messages()
            summary = self.summarize(messages)
            if not self.summary_rejected:
                self.conversation.messages_since_summary = 0
            self.conversation.summary["content"] = summary
        self.conversation.save_conversation()



class ContextWindowBuilder():
    def __init__(self, conversation, content_tokens, memory_tokens, min_messages):
        self.conversation = conversation
        self.content_tokens = content_tokens
        self.memory_tokens = memory_tokens
        self.min_messages = min_messages
        self.vector_memory = None

    def set_conversation(self, conversation):
        self.conversation = conversation
        self.vector_memory = embedder.Memory(self.conversation.memory_file)
        if not self.vector_memory.memory_exists():
            print("no memory")
            self.memorize_messages(self.conversation.messages)

    def memorize_message(self, message):
        summary_points = client.summarize_conversation([message])
        summary_points = [s for s in summary_points.split("\n") if s.startswith("- ")]
        for s in summary_points:
            print(s)
            self.vector_memory.encode_text(s)

    def memorize_messages(self, conversation):
        print("memorize_messages")
        for message in conversation:
            self.memorize_message(message)

    def build_notes_message(self, messages):
        print("building notes")
        if not self.vector_memory.memory_exists():
            print("no memory")
            self.memorize_messages()

        if len(messages) == 0:
            return None
        
        key = messages[-1]["content"]
        result = self.vector_memory.query(key, k=20)
        print(result)
        
        notes_message = "Parts from previous conversation you remember:"
        for page in result:
            note = page.page_content
            print(note)
            if any(note in m["content"] for m in messages):
                continue
            if note in notes_message:
                continue
            notes_message += f"\n\n{note}"
            if count_tokens(notes_message) >= self.memory_tokens:
                break

        print("notes built")
        return notes_message

    def last_message_index(self, messages=None):
        if messages is None:
            messages = self.conversation.messages
        index = -min([len(messages), 200])
        while num_tokens_from_messages(messages[index:]) > self.content_tokens:
            index=index+1
        if index > -self.min_messages:
            index = -self.min_messages
        print(f"{-index} messages with {num_tokens_from_messages(messages[index:])} tokens (max {self.content_tokens})")
        return index

    def out_of_context_messages(self):
        return list(self.conversation.messages[:self.last_message_index()])

    def in_context_messages(self, messages=None):
        """ Get the latest messages that fit in the token limit """
        if messages is None:
            messages = self.conversation.messages
        index = self.last_message_index(messages)
        short = messages[index:]
        
        summary = f"This is a summary you wrote for yourself: {self.conversation.summary['content']}"
        notes = self.build_notes_message(short)
        if notes:
            short.insert(0, {"role": "system", "content": notes})
        short.insert(0, {"role": "system", "content": summary})
        short.insert(0, {"role": "system", "content": self.conversation.system_message})
        return short



class LongChat():
    def __init__(
        self,
        conversation = "default.json",
        model = "gpt-4o",
        summarize_every = 5,
        summary_similarity_threshold = 0.2,
        content_tokens = client.content_tokens,
        memory_tokens = 500,
        summary_tokens = 300,
        reply_tokens = 2000,
        min_messages = 2,
        conversations_path = "conversations/"
    ):
        self.messages_since_summary = 0
        self.summarize_every = summarize_every
        self.summary_rejected = False
        self.model = model
        self.summary_prompt = """Provide a short but complete summary of our current conversation, including topics covered, key takeaways and conclusions? This summary is for you, and does not need to be human readable. Make only small updates to the previous summary to maintain coherence and relevance. The summary must be less than 200 words."""
        self.reply_tokens = reply_tokens
        self.conversations_path = conversations_path

        self.conversation = ConversationManager(conversation, self.conversations_path)
        self.summarizer = Summarizer(
            model, summary_tokens, summarize_every, summary_similarity_threshold,
            self.conversation
        )
        self.context = ContextWindowBuilder(self.conversation, content_tokens, memory_tokens, min_messages)


    def read_conversation(self, conversation):
        self.conversation = ConversationManager(conversation, self.conversations_path)
        self.summarizer.set_conversation(self.conversation)
        self.context.set_conversation(self.conversation)

    def in_context_messages(self):
        return self.context.in_context_messages()

    def out_of_context_messages(self):
        return self.context.out_of_context_messages()

    def post_user_message(self, user_message):
        if user_message != "":
            self.conversation.add_message(role = "user", content = user_message)
            self.context.memorize_message({"role": "user", "content": user_message})
    

    def request_ai_message(self,):
        in_context_messages = copy.deepcopy(self.in_context_messages())
        print(f"sending {len(in_context_messages)} messages with {num_tokens_from_messages(in_context_messages)} tokens")

        in_context_messages[-1]["content"] += """

Reply in the following format:
Scratchpad:
These are your internal thoughts and are not displayed to the user. Use this section to plan your reply and structure your thoughts.

Actual Reply:
This section should contain the actual reply to the user. This is what the user will see.
"""

        message = client.request_message(messages=in_context_messages,
        max_tokens = self.reply_tokens)
        
        parts = message.split("Actual Reply:")
        print(parts[0])
        message = parts[-1].strip().lstrip("*").strip().lstrip("#").strip()

        self.conversation.add_message(role = "assistant", content = message)
        self.context.memorize_message({"role": "assistant", "content": message})
        
        self.summarizer.check_summary(self.context)
        return {}
    

