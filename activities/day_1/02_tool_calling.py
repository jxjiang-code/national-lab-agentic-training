import argparse
import inspect
import json
from typing import Any
from typing import Callable

from openai import OpenAI
from pydantic import TypeAdapter

# TODO: Rename tool call
def tool_call() -> str:
    '''TODO: Write your documentation here
    '''

    # TODO: Write your tool call here


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

    tools = [generate_tool_schema(tool_call)]
    # Print the schema to see what the model is told about your tool
    print(json.dumps(tools, indent=2))

    input_items = [{'role': 'user', 'content': query}]
    response = client.responses.create(
        model=model,
        instructions='You are a helpful assistant.',
        input=input_items,
        tools=tools,
    )

    # TODO: Extract tool calls from `response.output` and run the call
    # Print what the model returned first to see the format

    # TODO: Append `response.output` to `input_items`, followed by 
    # one item per tool call:
    #   {'type': 'function_call_output', 'call_id': ..., 'output': ...}
    input_items += ...
    
    response = ... # TODO: Send the result back to the model

    print(response.output_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    parser.add_argument('query')
    ops = parser.parse_args()

    main(ops.model, ops.query)
