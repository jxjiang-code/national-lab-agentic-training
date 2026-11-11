import os
import getpass
from dotenv import load_dotenv

load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"var: ")

_set_env("OPENAI_API_KEY")


print("OPENAI_BASE_URL:", os.environ.get("OPENAI_BASE_URL"))
key = os.environ.get("OPENAI_API_KEY")
print("OPENAI_API_KEY set:", bool(key))
print("OPENAI_API_KEY preview:", f"{key[:6]}...{key[-4:]}" if key else None)

