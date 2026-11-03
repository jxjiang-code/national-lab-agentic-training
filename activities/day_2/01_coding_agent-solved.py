import argparse
import inspect
from pathlib import Path
import subprocess
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
    file = Path(path)
    if old_content == '':
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('a') as f:
            f.write(new_content)
        return True

    content = file.read_text()
    if old_content not in content:
        raise ValueError(f"old_content not found in {path}. Read the file to check the exact text.")
    file.write_text(content.replace(old_content, new_content, 1))
    return True

def run_command(cmd: str, timeout: float = 10) -> dict[str]:
    '''Runs a shell command and captures the output
    
    Args:
        cmd: Command to run. Can be a full shell command.
        timeout: how long the command can run before automatically terminating it.
    
    Returns: The return code, stdout and stderr as a dictionary.
    '''


    result = subprocess.run(cmd, capture_output=True, shell=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return {
        'cmd': cmd,
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
    }

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


def coding_agent_loop(model: str):
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
        while True:
            assistant_response = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=conversation,
                tools=tools,
            )
            conversation.extend(assistant_response.output)

            tool_calls = extract_tool_calls(assistant_response)

            for tool_call in tool_calls:
                print(f"{SYSTEM_COLOR}Tool call: {tool_call.name}({tool_call.arguments}){RESET_COLOR}")
                tool_output = execute_tool_call(tool_call)
                conversation.append(tool_output)

            if len(tool_calls) == 0:
                print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR}: {assistant_response.output_text}")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    ops = parser.parse_args()
    coding_agent_loop(ops.model)