from flask import Flask, render_template, request
from longchat import LongChat
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import html
import markdown
import re
from html import escape, unescape
import os

app = Flask(__name__)

conversations_path = "conversations/"


chatbot = LongChat()

def render_with_markdown(markdown_str):
    markdown_str = escape(markdown_str)
    code_blocks = re.findall(r'```(.*?)\n```', markdown_str, re.DOTALL)
    for code_block in code_blocks:
        code_lines = code_block.split("\n")[1:]
        language = code_block.split("\n")[0].strip()

        if not language:
            lexer = TextLexer()
        else:
            lexer = get_lexer_by_name(language, stripall=True)

        formatter = html.HtmlFormatter(style="colorful", noclasses=True)

        code_lines = '\n'.join([unescape(code_line) for code_line in code_lines])
        code_lines = code_lines.replace("#x27;", "'")
        highlighted_code = highlight(code_lines, lexer, formatter)
        if highlighted_code:
            replacement = f'<div class="card my-2"><div class="card-header">{language}</div><div class="card-body"><pre><code class="code">{highlighted_code}</code></pre></div></div>'
        else:
            replacement = ''

        markdown_str = markdown_str.replace(f'```{code_block}\n```', replacement)

    return markdown.markdown(markdown_str)


def render_messages(messages):
    rendered_messages = []
    for message in messages:
        try:
            rendered_content = render_with_markdown(message["content"])
        except:
            rendered_content = escape(message["content"])
        rendered_messages.append({
            "role": message["role"],
            "content": rendered_content,
            "is_summary": message["is_summary"]
        })
    return rendered_messages


def get_display_messages():
    print(f"displaying {len(chatbot.new_messages())} new messages")
    new_messages = [dict(m, is_summary=False) for m in chatbot.new_messages()]
    old_messages = [dict(m, is_summary=False) for m in chatbot.old_messages()]

    new_messages[0]["is_system"] = True
    new_messages[1]["is_summary_prompt"] = True
    new_messages[2]["is_summary"] = True

    return old_messages + new_messages


error_in = ""

@app.route("/", methods=["GET", "POST"])
def home():
    global error_in
    conversations = []
    for filename in os.listdir(conversations_path):
        conversations.append(filename)

    conversation_id = request.args.get("conversation_id")
    if not conversation_id and request.method == "POST":
        conversation_id = request.form["conversation_id"]
    if conversation_id:
        chatbot.read_conversation(conversation_id)
    else:
        chatbot.read_conversation(chatbot.conversation)

    if request.method == "POST":
        new_msg = request.form["new_message"]
        if new_msg:
            try:
                chatbot.new_message(new_msg)
                error_in = ""
            except Exception as e:
                print(e)
                error_in = new_msg
    
    display_messages = get_display_messages()
    rendered_messages = render_messages(display_messages)
    return render_template(
        "index.html",
        messages=rendered_messages,
        conversations=conversations,
        active_conversation_id=chatbot.conversation,
        error_in=error_in 
    )



if __name__ == "__main__":
    app.run(debug=True)

