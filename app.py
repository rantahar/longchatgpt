from flask import Flask, render_template, request, redirect, url_for
from longchat import LongChat
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import html
import traceback
import re
from html import escape, unescape
import os
import json

app = Flask(__name__)

conversations_path = "conversations/"


chatbot = LongChat()

def render_with_markdown(message_str):
    message_str = escape(message_str)
    code_blocks = re.findall(r'```(.*?)\n```', message_str, re.DOTALL)
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

        message_str = message_str.replace(f'```{code_block}\n```', replacement)
    message_str = message_str.replace('\n', '<br>')

    return message_str


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


def save_config():
    with open("config.json", 'w') as outfile:
        json.dump({
            "conversation": chatbot.conversation
        }, outfile, indent=4)


def load_config():
    try:
        with open("config.json", 'r') as infile:
            config = json.load(infile)
    except:
        config = {
            "conversation": "default"
        }
    chatbot.read_conversation(config["conversation"])


def get_display_messages():
    print(f"displaying {len(chatbot.new_messages())} new messages")
    new_messages = [dict(m, is_summary=False) for m in chatbot.new_messages()]
    old_messages = [dict(m, is_summary=False) for m in chatbot.old_messages()]

    new_messages[0]["is_summary"] = True
    new_messages[1]["is_summary"] = True

    return old_messages + new_messages


error_in = ""

@app.route("/", methods=["GET"])
def home():
    global error_in
    load_config()

    conversations = []
    for filename in os.listdir(conversations_path):
        conversations.append(filename)

    display_messages = get_display_messages()
    rendered_messages = render_messages(display_messages)

    return render_template(
        "index.html",
        messages=rendered_messages,
        conversations=conversations,
        active_conversation_id=chatbot.conversation,
        error_in=error_in,
        system_message=chatbot.system_message,
        disable_function_calls=chatbot.disable_functions
    )


@app.route("/select_conversation", methods=["GET"])
def select_conversation():
    global error_in
    load_config()

    conversation_id = request.args.get("conversation_id")
    if conversation_id:
        chatbot.read_conversation(conversation_id)
    else:
        chatbot.read_conversation(chatbot.conversation)

    save_config()
    return redirect(url_for("home"))


@app.route("/new_message", methods=["POST"])
def new_message():
    global error_in
    load_config()

    if request.method == "POST":
        if "new_message" in request.form:
            new_msg = request.form["new_message"]
            chatbot.post_user_message(new_msg)
    return redirect(url_for("home"))


@app.route("/request_ai_message", methods=["POST"])
def request_ai_message():
    global error_in
    load_config()

    if request.method == "POST":
        print(request.form)
        disable_function_calls = request.form.get("disable-function-calls", False)
        try:
            result = chatbot.request_ai_message(disable_function_calls=disable_function_calls)
            if "tool_calls" in result:
                return redirect(url_for("confirm_function_call", function_call=result["tool_calls"]))
            else:
                error_in = ""

        except Exception as e:
            print(traceback.format_exc())
            print(e)
    return redirect(url_for("home"))


@app.route("/confirm_function_call", methods=["GET", "POST"])
def confirm_function_call():
    global error_in

    load_config()
    if request.method == "POST":
        confirmed = request.form.get("confirmed")
        if confirmed == "yes":
            try:
                # Execute the function call
                result = chatbot.handle_function_call(json.loads(request.form.get("function_call")))
                print(result)
                if result.tool_calls:
                    return redirect(url_for("confirm_function_call", function_call=result.tool_calls))
                else:
                    error_in = ""
                    return redirect(url_for("home"))
            except Exception as e:
                print(e)
                print(traceback.format_exc())
        else:
            return redirect(url_for("home"))
    
    function_call = request.args.get("function_call")
    return render_template("confirmation.html", function_call=function_call)


@app.route("/save_system_message", methods=["POST"])
def save_system_message():
    load_config()

    if request.method == "POST":
        print(request.form)
        system_message = request.form["system_message"]
        chatbot.system_message = system_message
        chatbot.dump_conversation()
    return redirect(url_for("home", **request.args))


if __name__ == "__main__":
    app.run(debug=True)

