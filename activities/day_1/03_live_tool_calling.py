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
    print("Arms")
    return "Arms"

def add_numbers(a: float, b: float) -> str:
    '''Add two numbers together and return their exact sum.
    '''

    print(f"[真的执行了 Python] add_numbers(a={a}, b={b})")
    return str(a + b)

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

    tools = [
        generate_tool_schema(get_user_name),
        generate_tool_schema(add_numbers),
    ]
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
    print('input_tokens:', response.usage.input_tokens)
    print('output_tokens:', response.usage.output_tokens)
    print("<"+20*"-"+"第一次调用结束"+20*"-"+">")

    for item in response.output:
        print('items in response.output: ', item)
        if isinstance(item, ResponseFunctionToolCall):
            print('items is ResponseFunctionToolCall: ', item)
            if item.name == 'get_user_name':
                name = get_user_name(**json.loads(item.arguments))
                tool_return = { 
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": name
                }
                input_items.append(tool_return)
            elif item.name == 'add_numbers':
                result = add_numbers(**json.loads(item.arguments))
                tool_return = { 
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result
                }

    # 注意：response.output 已经在第 64 行 extend 过了，这里绝不能再 extend 一次。
    # input_items 是「完整对话记录」，重复 extend 会让同一个 call_id 出现两次。
    # 正确顺序由第 64 行的 extend + 循环里的 append 自然保证：
    #   [user, function_call(id=X), function_call_output(id=X)]
    print(input_items)
    print("<"+20*"-"+"调用Tool结束"+20*"-"+">")

    # response = ... # TODO: Send the result back to the model
    response = client.responses.create(
        model=model,
        instructions='You are a helpful assistant.',
        input=input_items,
        tools=tools,
    )

    print(response.output)
    print('input_tokens:', response.usage.input_tokens)
    print('output_tokens:', response.usage.output_tokens)
    print("<"+20*"-"+"第二次调用结束"+20*"-"+">")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    parser.add_argument('query')
    ops = parser.parse_args()

    main(ops.model, ops.query)
