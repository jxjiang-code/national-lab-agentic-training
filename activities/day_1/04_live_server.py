from fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP('day1-server')

# TODO: Add tools to the server with the @mcp.tool decorator

WORKSPACE = (Path(__file__).parent / "workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)

def _safe(path: str) -> Path:
    """Resolve a relative path inside WORKSPACE, blocking traversal."""
    p = (WORKSPACE / path).resolve()
    if WORKSPACE not in p.parents and p != WORKSPACE:
        raise ValueError(f"Path escapes workspace: {path}")
    return p

@mcp.tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file in the workspace."""
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {path}"


@mcp.tool
def edit_file(path: str, old: str, new: str) -> str:
    """Replace an exact substring in an existing file."""
    p = _safe(path)
    text = p.read_text()
    if old not in text:
        return f"Error: substring not found in {path}"
    p.write_text(text.replace(old, new))
    return f"Edited {path}"


if __name__ == '__main__':
    mcp.run(transport='http', port=8765)
