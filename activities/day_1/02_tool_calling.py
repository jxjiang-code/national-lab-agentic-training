import argparse
import inspect
from typing import Callable
from typing import Any

from openai import OpenAI

# TODO: Rename tool call
def tool_call() -> str:
    '''TODO: Write your documentation here
    '''

    # TODO: Write your tool call here


def generate_tool_call_prompt(func: Callable[[Any], Any]):
    '''Generates description for model from docstring.
    '''

    return f"""
        Name: {func.__name__}
        Description: {func.__doc__}
        Signature: {inspect.signature(func)}
    """

def main(model: str, query: str):
    tool_prompt = generate_tool_call_prompt(tool_call) 
    prompt = \
f'''
System: You are a helpful assistant. You have access to the following tools:
{tool_prompt}

User: {query}
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')
    parser.add_argument('query')
    ops = parser.parse_args()

    main(ops.model, ops.query)