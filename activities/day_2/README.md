# Activities for Day 2

In Day 2, we will move from bespoke agentic workflows to coding agents. 
To demistify how coding agents work, we will first build a simple coding agent.
Then we will move to using commercial agents

For model access, use the OpenAI compatible API to the ALCF inference endpoints.
Full documentation can be found [here](https://docs.alcf.anl.gov/services/inference-endpoints/#2-authenticate).
To authenticate and set your api token, run these commands:

```
$ uvx alcf-tokens login inference
$ export OPENAI_BASE_URL=https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1
$ export OPENAI_API_KEY=$(uvx alcf-tokens get-token inference)
```

Tokens are valid for 48 hours, so re-run the last command if you start getting authentication errors.

All of the scripts default to the `beam-1.0-bf16` model on Minerva. Use `-m/--model` to try a different model.
The models available on each cluster are listed [here](https://docs.alcf.anl.gov/services/inference-endpoints/models/).
If you switch to a model on a different cluster (e.g. Sophia, Metis, or Tara), you also need to change `OPENAI_BASE_URL` to that cluster's endpoint.

All of the commands below should be run from this directory (`activities/day_2`).

## Activity 01: Building a Coding Agent.

This activity will use the mechanisms from day 1 to build a coding agent.
This acitivity is inspired by this [guide](https://www.mihaileric.com/The-Emperor-Has-No-Clothes/), but slightly modified for an OpenAI compatible API and modern, server-side tool parsing.
The coding agent uses the same native tool calling from the previous class.
However, instead of implementing a bespoke tool, we will give the agent access to the commandline.

You must write the `run_commands` tool to allow the agent to run generic shell commands, and the tool calling loop inside the coding agent loop.
We purposely use commands that were close to the raw model calls to understand how the internals of a coding agent work.
After you are done, ask it to do two different tasks:

(1) Ask it to write a `hello_world.py` program to test if all the functionality works

(2) Find a paper on arxiv, produce multiple reivews of the paper, and the a unified score.

How well does the shell work compared to bespoke tools? Can the coding agent use existing APIs without an MCP server?
Right now we implemented manual approval on every tool call, how would you relax this constraint?

Extensions if time: 

 - Store and re-use the transcripts of previous sessions
 -  Implement context management and summarization