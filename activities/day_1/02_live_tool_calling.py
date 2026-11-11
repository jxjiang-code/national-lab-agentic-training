import argparse
import inspect
import json
from typing import Any
from typing import Callable

from openai import OpenAI
from openai.types.responses import ResponseFunctionToolCall
from pydantic import TypeAdapter

# TODO: Rename tool call
def tool_call() -> str:
    '''TODO: Write your documentation here
    '''

    # TODO: Write your tool call here
    print("This is a tool calling.")

def get_user_name() -> str:
    '''Return the name of the user running the code.
    '''

    # TODO: Write your tool call here
    print("Arm")

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


def main(model: str, query: str):
    client = OpenAI()

    tools = [generate_tool_schema(get_user_name)]
    # Print the schema to see what the model is told about your tool
    print(json.dumps(tools, indent=2))

    input_items = [{'role': 'user', 'content': query}]
    response = client.responses.create(
        model=model,
        instructions='You are a helpful assistant.',
        input=input_items,
        tools=tools,
    )
    input_items.extend(response.output)

    # TODO: Extract tool calls from `response.output` and run the call
    # Print what the model returned first to see the format
    print(response.output)

    for item in response.output:
        if isinstance(item, ResponseFunctionToolCall):
            print(item)
            if item.name == 'get_user_name':
                name = get_user_name(**json.loads(item.arguments))
                tool_return = { 
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": name
                }
                input_items.append(tool_return)


    # TODO: Append `response.output` to `input_items`, followed by 
    # one item per tool call:
    #   {'type': 'function_call_output', 'call_id': ..., 'output': ...}
    input_items.extend(response.output)
    
    # response = ... # TODO: Send the result back to the model
    response = client.responses.create(
        model=model,
        instructions='You are a helpful assistant.',
        input=input_items,
        tools=tools,
    )

    print(response.output_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    parser.add_argument('query')
    ops = parser.parse_args()

    main(ops.model, ops.query)
