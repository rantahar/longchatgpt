import json
import openai
from tokens import num_tokens_from_messages

messages_file = "messages.json"
max_summary_length = 120
model = "gpt-3.5-turbo"
summarize_every = 4

with open('api_key', 'r') as file1:
    openai.api_key = file1.readlines()[0].strip()

summary_prompt = "Please provide a detailed summary of the entire conversation so far. Summarize the main ideas, themes and topics."

with open('messages.json', 'r') as f:
    messages = json.load(f)

messages.insert(0, {"role": "system", "content": "You are a helpful assistant. You will occationally summarize the conversation for yourself when prompted. You will use existing summaries to continue the conversation naturally."})

# Define initial variables
current_message_index = 0
num_messages = len(messages)


def shorten_conversation(messages):
    short = messages[-20:]
    while num_tokens_from_messages(short) > 1024:
        short=short[1:]
        print("too long", num_tokens_from_messages(short))
    return short


def summarize_chatgpt(messages):
    print("summarizing")
    messages.append({"role": "user", "content": summary_prompt})
    result = openai.ChatCompletion.create(
      model=model, messages=shorten_conversation(messages)
    )
    messages.append({"role": "assistant", "content": result.choices[0].message.content})


def summarize_api(messages):
    print("summarizing")
    short = shorten_conversation(messages)
    messages_with_role = [f'{m["role"]}: {m["content"]}' for m in short]
    messages_text = '\ņ'.join(messages_with_role)
    summary = openai.Completion.create(
        engine="davinci",
        prompt=(f"{summary_prompt}:\n{messages_text}\n\nSummary:"),
        max_tokens=max_summary_length,
        temperature=0.5,
        n = 1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    ).choices[0].text.strip()
    messages.append({"role": "assistant", "content": summary})


index = len(messages)//2
def new_message(user_message):
    global index, messages
    messages.append({"role": "user", "content": user_message})

    try: 
        result = openai.ChatCompletion.create(
          model=model, messages=shorten_conversation(messages)
        )
    except Exception as e:
        print(e)
        messages = messages[:-1]
        return

    messages.append({"role": "assistant", "content": result.choices[0].message.content})

    with open(messages_file, 'w') as outfile:
        json.dump(messages[1:], outfile)

    if index % summarize_every == 0 and index > 0:
        try:
            summarize_chatgpt(messages)    
        except Exception as e:
            print(e)
            messages = messages[:-1]
    
        with open(messages_file, 'w') as outfile:
            json.dump(messages[1:], outfile)

    index += 1


