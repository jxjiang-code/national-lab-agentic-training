uv run python - <<'PY'
from pathlib import Path
from pypdf import PdfReader

for pdf in sorted(Path("sample_papers").rglob("*")):
    if pdf.suffix.lower() != ".pdf":
        continue
    print(f"\n--- {pdf} ---", flush=True)
    try:
        reader = PdfReader(str(pdf), strict=False)
        for page in reader.pages:
            page.extract_text()  # force page parsing
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
PY

