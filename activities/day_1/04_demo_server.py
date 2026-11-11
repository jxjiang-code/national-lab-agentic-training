from fastmcp import FastMCP
import random

mcp = FastMCP('day1-server')

# TODO: Add tools to the server with the @mcp.tool decorator
@mcp.tool
def moon_phase(input: str) -> str:
    '''Calcuates the phase of the mooon'''
    return "full"

# TODO: Add tools to the server with the @mcp.tool decorator
@mcp.tool
def estimate_pi(samples: int = 100_000) -> float:
   """Estimate pi by dropping random points in a square and counting how many land in the circle."""
   print("Estimating Pi %d" % samples)
   inside = sum(1 for _ in range(samples) if random.random() ** 2 + random.random() ** 2 <= 1)
   return 4 * inside / samples

if __name__ == '__main__':
    mcp.run(transport='http', port=8765)
