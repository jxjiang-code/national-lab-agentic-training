import asyncio
import argparse
import warnings


from langchain.agents import create_agent
from langchain_core._api import LangChainBetaWarning
from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore', category=LangChainBetaWarning)

from langchain.mcp import MCPAdapter


async def main(model: str, url: str) -> None:
    async with MCPAdapter(url) as adapter:
        tools = await adapter.list_tools()
        print(f'Connected to {url} with tools: {[t.name for t in tools]}')
        print('Type exit or quit to stop the conversation.')

        agent = create_agent(ChatOpenAI(model=model), tools=tools)
        messages = []

        while True:
            try:
                query = input('\nUser > ').strip()
            except EOFError:
                break
            if query in ('exit', 'quit'):
                break
            messages.append({'role': 'user', 'content': query})
            result = await agent.ainvoke({'messages': messages})
            messages = result['messages']
            print(messages[-1].content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    parser.add_argument('-u', '--url', default='http://localhost:8765/mcp')
    ops = parser.parse_args()

    asyncio.run(main(ops.model, ops.url))
