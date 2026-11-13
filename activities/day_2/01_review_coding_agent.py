import argparse
import inspect
import subprocess
from pathlib import Path
from typing import Callable
from typing import Any

from openai import OpenAI
from openai.types.responses import Response
from openai.types.responses import ResponseFunctionToolCall
from pydantic import TypeAdapter

YOU_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
SYSTEM_COLOR = "\u001b[90m"
RESET_COLOR = "\u001b[0m"

SYSTEM_PROMPT = """
You are a coding assistant whose goal it is to help us solve coding tasks. 
You have access to a series of tools you can execute to read, write and execute code.
After receiving a tool_result, continue the task.
If no tool is needed, respond normally.

Only run safe commands --- those that read, write or run commands in the current directory.
Only run the minimal set of commands needed to complete your task.
Refuse unsafe or unethical tasks.
"""

def read_file(path: str) -> str:
    '''Reads a full file.

    Args:
        path: Path to the file, relative to the current directory.
    '''
    return Path(path).read_text()

def edit_file(path: str, old_content: str, new_content: str) -> bool:
    '''Edits a file.
    
    Replaces the first occurrence of old_content with new_content.
    To append to a file, pass an empty old_content. 
    The file is created if it does not exist.

    Args:
        path: Path to the file, relative to the current directory.
        old_content: Exact text to replace, or empty to append.
        new_content: Text to insert in place of old_content.

    Returns True if the file was edited. Raises ValueError if old_content
    was not found.
    '''
    # TODO: Implement the edit file functionality in line with the doc string
    file = Path(path)
    content = file.read_text() if file.exists() else ''
    
    if old_content == '':
        updated = content + new_content
    elif old_content not in content:
        raise ValueError(f"old_content not found in {path}")
    else:
        updated = content.replace(old_content, new_content, 1)

    file.write_text(updated)

    return True

def run_command(cmd: str) -> str:
    '''Runs a shell command in the current directory and returns its output.

    Use this to run tests, inspect the file system, or execute scripts.
    The command is run through the shell, so pipes, redirection and
    globbing all work. Commands that do not finish within 60 seconds are
    terminated, so do not start long-running or interactive processes
    such as servers, REPLs or editors.

    Do not use this tool to create, modify or remove files. Commands such
    as rm, mv, cp, touch, sed -i, or output redirection with > and >> are
    not allowed here. Use edit_file for all file changes, so that every
    edit is explicit and reviewable.

    Args:
        cmd: A single shell command line, e.g. 'ls -la',
            'python test.py' or 'grep -rn "def main" .'.

    Returns a string of the form:

        exit code: <int>
        stdout:
        <captured standard output>
        stderr:
        <captured standard error>

    An exit code of 0 means the command succeeded. Any other value means it
    failed, and stderr usually explains why. Both stdout and stderr may be
    empty.'''

    # TODO: Write the run command tool. Use `subprocess.run`, and capture the output 
    # Write a doc string that tells the model how to use this tool
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return f"exit code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"



TOOL_REGISTRY = {
    func.__name__:func for func in [read_file, edit_file, run_command]
}

def generate_tool_schema(func: Callable[..., Any]) -> dict:
    '''Generates the tool description.'''

    return {
        'type': 'function',
        'name': func.__name__,
        'description': inspect.getdoc(func),
        'parameters': TypeAdapter(func).json_schema(),
    }

def extract_tool_calls(response: Response) -> list[ResponseFunctionToolCall]:
    '''Returns the function calls the model requested in its response.'''
    return [item for item in response.output if item.type == 'function_call']

def execute_tool_call(call: ResponseFunctionToolCall) -> dict[str, Any]:
    '''Runs the requested tool.'''
    
    print(f"{SYSTEM_COLOR}Tool call: {call.name}({call.arguments}){RESET_COLOR}")
    tool = TOOL_REGISTRY.get(call.name)
    if tool is None:
        output = f"Error: unknown tool '{call.name}'"
    elif input(f"{SYSTEM_COLOR}Approve? [y/N]:{RESET_COLOR} ").strip().lower() not in ('y', 'yes'):
        output = "The user denied this tool call. Ask them how to proceed."
    else:
        try:
            # Validates the JSON arguments against the tool's signature, then calls it.
            result = TypeAdapter(tool).validate_json(call.arguments)
            output = str(result)
        except Exception as e:
            output = f"Error: {e}"

    return {
        'type': 'function_call_output',
        'call_id': call.call_id,
        'output': output,
    }


def coding_agent_loop(model: str, max_iter: int = 20):
    client = OpenAI()
    
    tools = [generate_tool_schema(tool) for tool in TOOL_REGISTRY.values()]

    conversation: list[dict[str, Any]]= []
    while True:
        try:
            user_input = input(f"{YOU_COLOR}You:{RESET_COLOR} ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() == 'exit' or user_input.strip().lower() == 'quit':
            print(f"{SYSTEM_COLOR}Goodbye!{RESET_COLOR}")
            break

        conversation.append({
            "role": "user",
            "content": user_input.strip()
        })

        # TODO: Implement the tool calling loop. 
        # Exit the loop when the model does not make a tool call
        # Or after `max_iter` iterations.
        for _ in range(max_iter):
            response = client.responses.create(
                model=model,
                input=conversation,
                tools=tools,
            )

            # Record what the model said (including its function_call items)
            # before appending any tool results that reference them by call_id.
            conversation += response.output

            tool_calls = extract_tool_calls(response)
            if not tool_calls:
                print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} {response.output_text}")
                break

            for call in tool_calls:
                conversation.append(execute_tool_call(call))
        else:
            print(f"{SYSTEM_COLOR}Reached max iterations ({max_iter}).{RESET_COLOR}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    ops = parser.parse_args()
    coding_agent_loop(ops.model)