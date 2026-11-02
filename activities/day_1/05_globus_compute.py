import argparse
import random
from functools import wraps

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from globus_compute_sdk import Executor

compute_endpoint = None  # TODO: Fill in the endpoint id that you started

gce = Executor(endpoint_id=compute_endpoint)


def gce_wrapper(func):
    """Wrapper to execute the function via Globus Compute"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        future = gce.submit(func, *args, **kwargs)
        return future.result()
    return wrapper

@gce_wrapper
def compute_primes(n: int = 1000) -> dict:
    """Count prime numbers up to n."""
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True

    primes = [x for x in range(2, n + 1) if is_prime(x)]
    return {
        "task": "prime_count",
        "n": n,
        "count": len(primes),
        "largest": primes[-1] if primes else None,
        "first_10": primes[:10]
    }

@gce_wrapper
def compute_energy(molecule: str = "H2O") -> dict:
    """Simulate molecular energy calculation."""
    import random
    # Mock responses - real version would use PySCF or similar
    energies = {"H2O": -76.4, "CO2": -188.5, "CH4": -40.5,"N2": -109.5}
    base_energy = energies.get(molecule, -50.0) + random.uniform(-0.1, 0.1)
    return {
        "task": "energy_calculation",
        "molecule": molecule,
        "energy_hartree": base_energy,
        "method": "surrogate_dft",
        "basis": "cc-pVDZ"
    }

@gce_wrapper
def compute_trajectory(n_atoms: int = 27, n_steps: int = 100) -> dict:
    """Simulate MD trajectory."""
    import random
    temperatures = [300 + random.uniform(-20, 20) for _ in range(n_steps)]
    return {
        "task": "md_simulation",
        "n_atoms": n_atoms,
        "n_steps": n_steps,
        "avg_temperature": sum(temperatures) / len(temperatures),
        "final_temperature": temperatures[-1],
    }


requests=["How many prime numbers are there below 500?",
        "What is the potential energy of a water molecule",
        "What is the MD trajectory of water molecules"]

CHOOSER_PROMPT = """You are a task router. Given a user request, choose which computation to run.

Available computations:
1. "primes" - Count prime numbers (for math/number questions)
2. "energy" - Calculate molecular energy (for chemistry/molecule questions)
3. "trajectory" - Run MD simulation (for physics/dynamics/trajectory questions)
"""

def main(model: str):
    llm = ChatOpenAI(model=model)

    tools = [compute_primes, compute_energy, compute_trajectory]
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=CHOOSER_PROMPT,
    )

    # Run the agent
    request=random.choice(requests)

    print(f'User: {request}')

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": request}]},
        stream_mode='updates',
        version="v2"
    ):
        if chunk["type"] == "updates":
            for step, data in chunk["data"].items():
                print(f"step: {step}")
                print(f"content: {data['messages'][-1].content_blocks}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='beam-1.0-bf16')
    ops = parser.parse_args()

    main(ops.model)
