import os
import re

project_folder = ".."

def calculate(eval_string):
    result = eval(eval_string)
    return result, """success"""

def read_code_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    return code, f"""The file "{file_path}" was read successfully"""

def extract_function_definition(function_name):
    pattern = r"def " + function_name + r"\(.*?\):.*?(?=def|\Z)"
    function_definitions = []

    for root, dirs, files in os.walk(project_folder):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_folder)
                
                with open(file_path, "r") as f:
                    content = f.read()
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        function_definition = match.group(0)
                        function_definitions.append(f"in {relative_path}:\n{function_definition}")


    function_definitions = "```\n" + f"\n\n".join(function_definitions) + "\n```"

    return function_definitions, f"""The function definition "{function_name}" was read successfully"""


definitions = [
    {
        "name": "read_code_file",
        "description": "Read a code file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the code file"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "extract_function_definition",
        "description": "Extract a function definition",
        "parameters": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "The name of the function"
                }
            }
        }
    },
    {
        "name": "calculate",
        "description": "Do numerical calculation. For example providing '2*2' would return 4.",
        "parameters": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Expression to calculate. For example '2*2'"
                }
            }
        }
    }
]

implementations = {
    "read_code_file": read_code_file,
    "extract_function_definition": extract_function_definition,
    "calculate": calculate
}