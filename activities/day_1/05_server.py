from fastmcp import FastMCP

mcp = FastMCP('day1-server')

# TODO: Add tools to the server with the @mcp.tool decorator


if __name__ == '__main__':
    mcp.run(transport='http', port=8765)
