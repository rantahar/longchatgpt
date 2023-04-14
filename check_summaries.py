import json
from longchat import LongChat

chatbot = LongChat()
chatbot.summary = {"role": "assistant", "content":"We have not started the conversation yet."}


contexts = []
for n in range(1,10):
    messages = chatbot.messages[:2*n]
    print(messages)
    contexts += [chatbot.new_messages(messages)]

    chatbot.summarize_chatgpt(messages)    

    with open("check.json", 'w') as outfile:
        json.dump(contexts, outfile, indent=4)

contexts += [chatbot.new_messages(messages)]
with open("check.json", 'w') as outfile:
    json.dump(contexts, outfile, indent=4)

