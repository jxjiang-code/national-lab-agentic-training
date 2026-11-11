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


system_prompt = \
'''
You are taking part in an iterative scientific hypothesis review loop.
Be concrete and specific. Ground every statement in the context you are given,
and do not invent results, citations, or facts the context does not support.
'''

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

    approved: bool = Field(
        description=(
            'True only if the hypothesis has no major issues and is ready '
            'to stand as-is. False if any issue needs to be addressed.'
        ),
    )
    issues: list[str] = Field(
        description=(
            'Concrete, actionable issues the author must address. '
            'Empty if approved.'
        ),
    )
    score: int = Field(
        ge=1,
        le=10,
        description=(
            'Overall quality of the hypothesis from 1 (poor) to '
            '10 (excellent), judged on novelty, testability, feasibility, '
            'and grounding in the context.'
        ),
    )


class HypothesisState(TypedDict):
    '''Conversation state.'''
    context: str
    hypothesis: str
    iteration: int
    max_iterations: int
    # `operator.add` is the reducer: instead of overwriting `critiques` with
    # each node update, LangGraph computes `old + new`. For lists that is
    # concatenation, so the full review history survives the loop.
    critiques: Annotated[list[Critique], operator.add]


def build_graph(llm: ChatOpenAI):

    # Bind the schema once, outside the node, rather than on every call.
    # `with_structured_output` returns a *new* runnable that advertises the
    # `Critique` JSON schema to the model and parses the reply back into a
    # `Critique` instance -- so the node gets an object, not a string.
    critic_llm = llm.with_structured_output(Critique)

    def propose(state: HypothesisState) -> dict:
        '''Generate the initial hypothesis from the context.'''
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=propose_prompt.format(context=state['context'])),
        ]
        response = llm.invoke(messages)
        return {'hypothesis': response.content}

    def critique(state: HypothesisState) -> dict:
        '''Review the current hypothesis and return structured feedback.'''
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=critique_prompt.format(
                    context=state['context'],
                    hypothesis=state['hypothesis'],
                ),
            ),
        ]
        result = critic_llm.invoke(messages)
        # The update is a *list* because the reducer concatenates it onto the
        # history. Returning the bare object would raise a TypeError.
        return {
            'critiques': [result],
            'iteration': state['iteration'] + 1,
        }

    def revise(state: HypothesisState) -> dict:
        '''Rewrite the hypothesis to address the most recent critique.'''
        last_critique = state['critiques'][-1]
        issues = '\n'.join(f'- {issue}' for issue in last_critique.issues)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=revise_prompt.format(
                    context=state['context'],
                    hypothesis=state['hypothesis'],
                    issues=issues,
                ),
            ),
        ]
        response = llm.invoke(messages)
        return {'hypothesis': response.content}

    def should_continue(state: HypothesisState) -> str:
        '''Router: decide where to go after a critique.'''
        last_critique = state['critiques'][-1]
        if last_critique.approved:
            return 'end'
        if state['iteration'] >= state['max_iterations']:
            return 'end'
        return 'revise'

    graph = StateGraph(HypothesisState)
    graph.add_node('propose', propose)
    graph.add_node('critique', critique)
    graph.add_node('revise', revise)

    graph.add_edge(START, 'propose')
    graph.add_edge('propose', 'critique')
    graph.add_conditional_edges(
        'critique',
        should_continue,
        {'revise': 'revise', 'end': END},
    )
    # The back edge that closes the loop: every revision gets re-reviewed.
    graph.add_edge('revise', 'critique')

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
