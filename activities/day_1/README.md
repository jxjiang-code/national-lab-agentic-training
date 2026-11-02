# Activities for Day 1

In Day 1, we will investigate model infernce, tool-calling, and MCP servers to (1) build up the mechanisms that underly coding agents like Claude Code and Pi, and (2) learn how we use agents to interact with DOE infrastrcture.


For model access, use the OpenAI compatible API to the ALCF inference endpoints.
Full documentation can be found [here](https://docs.alcf.anl.gov/services/inference-endpoints/#2-authenticate).
To authenticate and set your api token, run these commands:

```
uvx alcf-tokens login inference
export OPENAI_API_KEY=$(alcf-tokens get-token inference)
```

## Activity 00: Extract Metadata Information From Abstract
---

In this activity, connect a OpenAI python client with the ALCF inference server.
Some starter code is provided in `00_extract_claims.py`. Use the openAI client to extract claims from the abstract.
You can run the code with `uv`:

```
uv run python 00_extract_claims.py
```

Try putting in an abstract or paper that you understand so you can validate the results. 
Vary the model and the temperature to investigate how those parameters effect the results coming out of the model. How are the models performing? How could you measure the performance? Could you scale the validation up?

## Activity 01: Self-Consistency and Tree-of-Thought
---
Now, use agents to review a paper. Build the self-consistency pattern and the tree-of-thought pattern to review a paper (A sample paper is included in the repo, but you may also use the text from your own paper, or a paper in your field.)
The code is written with `asyncio` so that you can run multiple model queries in parallel.
Does the review you get from a tree-of-thought or using self-consistency seem better?
How could we evaluate the review? Should these reviews be aligned with human reviews?


## Activity 02: Tool Calling
---
Next we will look at the internals of how agents act by writing enabling an agent to use a tool.

First think of an ability that would be useful for an agent in your domain.
That could be access to some database for information that is not part of the model's parametric knowledge, it could be the abiity to run a simulation, etc.
Implement that functionality in a python function.

Then write the logic to extract the tool calls from a models prompt. Once you tell the model about the tool, the inference service parses the model's raw text response into separate list. 
Your code should read that list and execute the tool call with the arguments provided.
You can try requests to the model to seed what format the response is in to decide how to parse the call.

## Activity 03: Use Agentic Frameworks
---

This activity will use LangGraph and PydanticAI to build patterns agent interaction. 
F

## Activity 04: Build A MCP Server
---

##  Activity 05: Globus Compute