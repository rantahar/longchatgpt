from flask import Flask, render_template, request
from longchat import messages, new_message
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import html
import markdown
import re
import bleach
from html import escape, unescape

app = Flask(__name__)

def render_message_content(markdown_str):
    markdown_str = escape(markdown_str)
    code_blocks = re.findall(r'```(.*?)\n```', markdown_str, re.DOTALL)
    for code_block in code_blocks:
        code_lines = code_block.split("\n")[1:]
        language = code_block.split("\n")[0].strip()

        if not language:
            lexer = TextLexer()
        else:
            lexer = get_lexer_by_name(language, stripall=True)

        formatter = html.HtmlFormatter(style="colorful")

        code_lines = ''.join([unescape(code_line) for code_line in code_lines])
        print(code_lines)
        code_lines = code_lines.replace("#x27;", "'")
        highlighted_code = highlight(code_lines, lexer, formatter)
        print(highlighted_code)
        if highlighted_code:
            replacement = f'<div class="card my-2"><div class="card-header">{language}</div><div class="card-body"><pre><code class="code">{highlighted_code}</code></pre></div></div>'
        else:
            replacement = ''
        print("replacement", replacement)

        markdown_str = markdown_str.replace(f'```{code_block}\n```', replacement)

    return markdown.markdown(markdown_str)

def render_messages(messages):
    rendered_messages = []
    for message in messages:
        rendered_content = render_message_content(message["content"])
        rendered_messages.append({"role": message["role"], "content": rendered_content})
    return rendered_messages

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        new_msg = request.form["new_message"]
        if new_msg:
            new_message(new_msg)

    rendered_messages = render_messages(messages)
    return render_template("index.html", messages=rendered_messages)




if __name__ == "__main__":
    app.run(debug=True)

