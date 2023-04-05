import os
import openai

model = "gpt-3.5-turbo"

with open('api_key', 'r') as file1:
    openai.api_key = file1.readlines()[0].strip()

summary_prompt = "Please provide a detailed summary of the entire conversation so far. Summarize the main ideas, themes and topics."

summary = """Sure, I\'d be happy to summarize our conversation so far:\n\nWe started by discussing the development of a chat-bot with long-term memory capabilities that periodically summarizes the conversation using the OpenAI GPT API. We then moved on to talking about interface options for displaying the chat-bot messages and code snippets, exploring potential libraries including PyQt5, Flask, FastAPI, and Streamlit.\n\nWe then discussed the limitations of textual summarization and the possibility of incorporating sentiment analysis to better capture the mood and context of the conversation. We also talked about the challenges associated with accurately capturing the nuances of a conversation with sentiment analysis.\n\nLater on, we discussed the development of a Flask application and how to display the current "message" variable in the main view. We also talked about the issue of ambiguity in the summarization prompt for the chat-bot and how to improve it by specifying the range of steps to be included in the summary.\n\nOverall, our conversation touched on a variety of topics related to chat-bot development, including long-term memory capabilities, interface options, summarization techniques, sentiment analysis, and more.

``` python
    code_blocks = re.findall(r'```(.*?)```', markdown_str, re.DOTALL)
```
"""


messages=[
    {"role": "system", "content": "You are a helpful assistant. You will occationally summarize the conversation for yourself when prompted. You will use existing summaries to continue the conversation naturally."},
    {"role": "user", "content": summary_prompt},
    {"role": "assistant", "content": summary},
]


summarize_every = 5

index = 1
def new_message(user_message):
    global index
    messages.append({"role": "user", "content": user_message})

    result = openai.ChatCompletion.create(
      model=model, messages=messages
    )
    print(result)

    messages.append({"role": "assistant", "content": result.choices[0].message.content})

    if index % summarize_every == 0:
        result = openai.ChatCompletion.create(
          model=model, messages=messages
        )

        messages.append({"role": "user", "content": summary_prompt})
        messages.append({"role": "assistant", "content": result.choices[0].message.content})
    
    index += 1
    print(messages)


