import argparse

from openai import OpenAI
import os

#import argo-openai

def main(model: str, abstract: str):
    with open(abstract) as fp:
        text = fp.read()
    client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                    api_key=os.environ["OPENAI_API_KEY"]
                    )

    prompt = f'Extract the claims and evidence from this paper:\n {text}'

    # TODO: Use the openAI client to extract claims from the abstract
    # Try putting in an abstract or paper that you understand so you
    # can validate the results. 
    # Vary the model and the temperature to investigate how those parameters
    # effect the results coming out of the model.
    # How are the models performing? How could you measure the performance?
    # Could you scale the validation up?

    response = client.responses.create(
        model=model,
        input=prompt)

    #!print(response)
    print(response.output_text)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    parser.add_argument('-a', '--abstract', default='sample_abstract.txt')
    ops = parser.parse_args()
    #print(ops.model, ops.abstract)

    main(ops.model, ops.abstract)
