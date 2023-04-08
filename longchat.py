import json
import openai
from tokens import num_tokens_from_messages

messages_file = "messages.json"

model = "gpt-3.5-turbo"

with open('api_key', 'r') as file1:
    openai.api_key = file1.readlines()[0].strip()

summary_prompt = "Please provide a detailed summary of the entire conversation so far. Summarize the main ideas, themes and topics."

with open('messages.json', 'r') as f:
    messages = json.load(f)

messages.insert(0, {"role": "system", "content": "You are a helpful assistant. You will occationally summarize the conversation for yourself when prompted. You will use existing summaries to continue the conversation naturally."})

# Define initial variables
current_message_index = 0
num_messages = len(messages)


summarize_every = 5

index = len(messages)//2
def new_message(user_message):
    global index
    messages.append({"role": "user", "content": user_message})

    send_messages = messages[-20:]
    while num_tokens_from_messages(send_messages) > 1024:
         send_messages=send_messages[1:]
         print("too long", num_tokens_from_messages(send_messages))

    result = openai.ChatCompletion.create(
      model=model, messages=send_messages
    )

    messages.append({"role": "assistant", "content": result.choices[0].message.content})

    if index % summarize_every == 0:
        send_messages.append({"role": "assistant", "content": result.choices[0].message.content})
        send_messages.append({"role": "user", "content": summary_prompt})
        result = openai.ChatCompletion.create(
          model=model, messages=send_messages
        )

        messages.append({"role": "user", "content": summary_prompt})
        messages.append({"role": "assistant", "content": result.choices[0].message.content})
    
    with open(messages_file, 'w') as outfile:
            json.dump(messages[1:], outfile)

    index += 1


