import argparse
import inspect
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
    '''Reads a file and returns its full contents as text.

    Args:
        path: Path to the file, relative to the current directory.
    '''
    return Path(path).read_text()

def edit_file(path: str, old_content: str, new_content: str) -> bool:
    '''Edits a file by replacing the first occurrence of old_content with
    new_content. To append to a file, pass an empty old_content; new_content
    is added to the end of the file, which is created if it does not exist.

    Args:
        path: Path to the file, relative to the current directory.
        old_content: Exact text to replace, or empty to append.
        new_content: Text to insert in place of old_content.

    Returns True if the file was edited. Raises ValueError if old_content
    was not found.
    '''
    # TODO: Implement the edit file functionality in line with the doc string

    return True

def run_command(cmd: str) -> str:
    '''Unimplemented.'''

    # TODO: Write the run command tool. Use `subprocess.run`, and capture the output 
    # Write a doc string that tells the model how to use this tool

    ...

TOOL_REGISTRY = {
    func.__name__:func for func in [read_file, edit_file, run_command]
}

def generate_tool_schema(func: Callable[..., Any]) -> dict:
    '''Generates the tool description for the model from the function's
    name, docstring, and type hints.
    '''

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
    '''Runs the requested tool and packages the result for the conversation.

    Every call must be approved by the user before it runs. Errors and
    denials are returned to the model as the tool output so it can recover.
    '''
    
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
            user_input = input(f"{YOU_COLOR}You:{RESET_COLOR}:")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() == 'exit' or user_input.strip().lower() == 'quit':
            print(f"{SYSTEM_COLOR}Goodbye!{RESET_COLOR}")
            break

        conversation.append({
            "role": "user",
            "content": user_input.strip()
        })

        # TODO: Implemet the tool calling loop. 
        # Exit the loop when the model does not make a tool call
        # Or after 20 iterations.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    ops = parser.parse_args()
    coding_agent_loop(ops.model)