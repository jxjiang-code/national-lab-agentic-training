import argparse
import asyncio

from openai import AsyncOpenAI
import os

review_prompt = \
'''
You are an expert academic computer scientist reviewing a vision piece for a journal.
Given the text of this paper, extract the claims and arguments to produce a concise summary. 
Then review the paper, producing a score with 1 being strong reject and 6 being strong accept.
Avoid borderline score 3. Conclude in the last line `Score: <1-6>`
"{text}"
'''

metareview_prompt = \
'''
Evaluate each of the reviews for accuracy, logical coherency, fairness and scholarly depth. Produce a
quality score for each of the reviews.

Reviews:
{reviews}
'''

score_prompt = \
'''
Unify the review scores and the quality scores to produce a final score for the paper. 
Produce a score with 1 being strong reject and 6 being strong accept.
Conclude in the last line `Score: <1-6>`

Reviews:
{reviews}

Quality scores:
{metareview}
'''


async def self_consistency(model: str, paper: str, n: int = 4):
    with open(paper) as fp:
        text = fp.read()
    client = AsyncOpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                         api_key=os.environ["OPENAI_API_KEY"]
                        )

    # TODO: Call the review prompt on the paper. Use async to parallelize the calls.
    #!review_futures = ...
    review_futures = [
    client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': review_prompt.format(text=text)}],
        temperature=0.9,
    )
    for _ in range(n)
    ]
    reviews = await asyncio.gather(*review_futures)

    # TODO: Use a majority vote or average to unify the scores
    import re

    review_texts = [r.choices[0].message.content for r in reviews]

    scores = []
    for t in review_texts:
        m = re.search(r'Score:\s*([1-6])', t)
        if m:
            scores.append(int(m.group(1)))

    print(f'Individual scores: {scores}')   # 先看一眼，很重要

    if not scores:
        raise ValueError('No scores could be parsed from any review')

    from collections import Counter
    final = Counter(scores).most_common(1)[0][0]    # 出现次数最多的那个分数
    print(final)



async def tree_of_thought(model: str, paper: str, n: int = 4):
    with open(paper) as fp:
        text = fp.read()
    client = AsyncOpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                         api_key=os.environ["OPENAI_API_KEY"]
                         )

    # TODO: Call the review prompt on the paper. Use async to parallelize the calls.
    #!review_futures = ...
    review_futures = [
        client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': review_prompt.format(text=text)}],
        temperature=0.7,
    )
    for _ in range(n)
    ]
    reviews = await asyncio.gather(*review_futures)

    # TODO: Call model with the metareview prompt
    review_texts = [r.choices[0].message.content for r in reviews]
    joined_reviews = '\n\n'.join(
        f'--- Review {i} ---\n{t}' for i, t in enumerate(review_texts, start=1)
        )
    metareview_response = await client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': metareview_prompt.format(reviews=joined_reviews)}],
        temperature=0.0,
        )
    metareview = metareview_response.choices[0].message.content

    # TODO: Call the model with the score prompt to produce a score
    score_response = await client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': score_prompt.format(
        reviews=joined_reviews,metareview=metareview,)
        }],
        temperature=0.0,
        )
    final = score_response.choices[0].message.content
    print(final)


STRATEGIES = {
    'self-consistency': self_consistency,
    'tree-of-thought': tree_of_thought,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    parser.add_argument('-p', '--paper', default='agency_as_paradigm.tex')
    parser.add_argument('-n', '--reps', type=int, default=4)
    parser.add_argument(
        '-s', '--strategy',
        choices=STRATEGIES.keys(),
        default='self-consistency',
        help='Reasoning pattern used to review the paper',
    )
    ops = parser.parse_args()

    strategy = STRATEGIES[ops.strategy]
    asyncio.run(strategy(ops.model, ops.paper, ops.reps))