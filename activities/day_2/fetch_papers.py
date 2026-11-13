"""One-off helper: download the first N MLSys 2026 papers into sample_papers/.

Usage:
    python3 -I fetch_papers.py 20
"""
import html
import re
import sys
import urllib.request
from pathlib import Path

INDEX = 'https://proceedings.mlsys.org/paper_files/paper/2026'
BASE = 'https://proceedings.mlsys.org'
OUT = Path(__file__).parent / 'sample_papers'


def slug(title: str) -> str:
    """Turns a paper title into a short, readable filename stem."""
    s = re.sub(r'[^a-zA-Z0-9]+', '_', title).strip('_').lower()
    return s[:60].rstrip('_')


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    page = urllib.request.urlopen(INDEX).read().decode('utf-8', 'replace')
    pairs = re.findall(r'href="(/paper_files/paper/2026/hash/([0-9a-f]+)-Abstract-Conference\.html)"[^>]*>(.*?)</a>',
                       page, re.S)

    OUT.mkdir(exist_ok=True)
    for i, (_, digest, raw_title) in enumerate(pairs[:n], 1):
        title = html.unescape(re.sub('<[^>]+>', '', raw_title)).strip()
        dest = OUT / f'{slug(title)}.pdf'
        if dest.exists():
            print(f'[{i}/{n}] skip {dest.name}')
            continue
        url = f'{BASE}/paper_files/paper/2026/file/{digest}-Paper-Conference.pdf'
        data = urllib.request.urlopen(url).read()
        dest.write_bytes(data)
        print(f'[{i}/{n}] {dest.name} ({len(data) // 1024} KB)')


if __name__ == '__main__':
    main()
