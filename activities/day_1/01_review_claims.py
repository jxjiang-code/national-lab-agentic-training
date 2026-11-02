import argparse
import asyncio

from openai import OpenAI

review_prompt = \
'''
You are an expert academic computer. scientist reviewing a vision piece for a journal.
Given the text of this paper, extract the claims and arguments to produce a conscise summary. 
Then review the paper, producing a score with 1 being strong reject and 6 being string accept.
Avoid borderline score 3. Conclude in the last line `Score: {1-6}`\n'
"{text}"
'''

metareview_prompt = \
'''
Evalute each of the reviews for accuracy, logical coherency, fairness and scholarly depth. Produce a
quality score for each of the reviews.
'''

score_prompt = \
'''
Unify the review scores and the quality scores to produce a final score for the paper. 
Produce a score with 1 being strong reject and 6 being string accept.
Conclude in the last line `Score: {1-6}`\n'

'''


async def main(model: str, paper: str, n: int = 4):
    with open(paper) as fp:
        text = fp.read()
    client = OpenAI()


    # TODO: Call the review prompt on the paper. Use async to parallelize the calls.
    review_futures = ...
    await asyncio.gather(review_futures)
    

    # TODO: Call model with the metareview prompt
    # TODO: Call the model with the score prompt to produce a score prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')
    parser.add_argument('-p', '--paper', default='agency_as_paradigm.tex')
    parser.add_argument('-n', '--reps', type=int, default=4)
    ops = parser.parse_args()

    asyncio.run(main(ops.model, ops.abstract, ops.reps))