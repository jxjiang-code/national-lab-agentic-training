import argparse
import operator
from typing import Annotated
from typing import TypedDict

from pydantic import BaseModel
from pydantic import Field

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph


propose_prompt = \
'''
You are a creative research scientist. Read the following research context and propose
a single novel, testable hypothesis that extends or challenges the work. State the hypothesis
in one or two sentences, then briefly describe an experiment that could test it.

Context:
"{context}"
'''

critique_prompt = \
'''
You are a skeptical reviewer on a grant panel. Critique the hypothesis below for novelty,
testability, feasibility, and grounding in the provided context. List concrete issues.
Only approve the hypothesis if it has no major issues.

Context:
"{context}"

Hypothesis:
"{hypothesis}"
'''

revise_prompt = \
'''
You are a research scientist revising your hypothesis in response to reviewer feedback.
Address every issue raised by the reviewer while keeping the hypothesis novel and testable.
Return only the revised hypothesis and the experiment that could test it.

Context:
"{context}"

Current hypothesis:
"{hypothesis}"

Reviewer issues:
{issues}
'''


class Critique(BaseModel):
    '''Feedback from the critic node.'''
    # TODO: Define the fields the critic should return. 
    # The router needs to know whether the hypothesis is approved, 
    # and the reviser needs the list of issues to address.
    ...


class HypothesisState(TypedDict):
    '''Conversation state.'''
    context: str
    hypothesis: str
    iteration: int
    max_iterations: int
    # TODO: Use `Annotated[..., operator.add]` so each critique node update
    # is appended to the list instead of replacing it.
    critiques: list[Critique]


def build_graph(llm: ChatOpenAI):

    def propose(state: HypothesisState) -> dict:
        '''Generate the initial hypothesis from the context.'''
        # TODO: Invoke the LLM prompt with the propose prompt
        # and update the conversation state.
        ...

    def critique(state: HypothesisState) -> dict:
        '''Review the current hypothesis and return structured feedback.'''
        # TODO: Bind the `Critique` schema to the llm so it returns a `Critique` 
        # object directly. Use `llm.with_structured_output`.
        ...

    def revise(state: HypothesisState) -> dict:
        '''Rewrite the hypothesis to address the most recent critique.'''
        # TODO: Invoke the LLM with the revise prompt.
        ...

    def should_continue(state: HypothesisState) -> str:
        '''Router: decide where to go after a critique.'''
        # TODO: Stop when the critic approves the hypothesis or we hit
        # `max_iterations`
        ...

    graph = StateGraph(HypothesisState)
    graph.add_node('propose', propose)
    graph.add_node('critique', critique)
    graph.add_node('revise', revise)

    # TODO: Wire up the propose -> critique -> (revise -> critique)* loop.
    #   1. Connect START to the first node.
    #   2. Add fixed edges with `graph.add_edge(src, dst)`.
    #   3. After `critique`, branch with `graph.add_conditional_edges`, using
    #      `should_continue` as the router and a path map from its return
    #      values ("revise" / "end") to node names (use END to finish).

    return graph.compile()


def main(model: str, context_file: str, max_iterations: int):
    with open(context_file) as fp:
        context = fp.read()

    llm = ChatOpenAI(model=model)
    app = build_graph(llm)

    # Print a diagram of the graph
    print(app.get_graph().draw_mermaid())

    initial_state = {
        'context': context,
        'hypothesis': '',
        'iteration': 0,
        'max_iterations': max_iterations,
        'critiques': [],
    }

    for chunk in app.stream(initial_state, stream_mode='updates'):
        for node, update in chunk.items():
            print(f'===== {node} =====')
            print(update)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    parser.add_argument('-c', '--context', default='sample_abstract.txt')
    parser.add_argument('-n', '--max-iterations', type=int, default=3)
    ops = parser.parse_args()

    main(ops.model, ops.context, ops.max_iterations)
