"""Long-context recall benchmark, with retrieval instead of a long context.

Same questions and judge as 03_long_context_benchmark.py, but instead of
pasting every paper into the prompt, the papers are chunked and ingested into
a vector database. The model gets a search_papers tool and must retrieve the
passages it needs.

The agent loop uses the OpenAI Agents SDK (openai-agents) instead of the
hand-written Responses API loop from 01_coding_agent.py.

The vector database is Chroma, running in memory. Chroma only stores and
searches the vectors; the embeddings themselves come from an embedding model
(google/embeddinggemma-300m) served by the ALCF inference service on Sophia.
Chat and judge models still use OPENAI_BASE_URL (e.g. Minerva).

Example:
    uv run python 04_long_context_with_rag.py \
        --papers sample_papers --questions questions.json --num-papers 10
"""
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from agents import (
    Agent,
    MaxTurnsExceeded,
    OpenAIResponsesModel,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    function_tool,
    set_tracing_disabled,
)
from openai import AsyncOpenAI, OpenAI
from pypdf import PdfReader

ANSWER_PROMPT = """
You are a research assistant answering questions about a collection of
academic papers. You cannot see the papers directly. Use the search_papers
tool to retrieve relevant passages; you may search several times with
different wording. Answer using only information from the retrieved passages.
If you cannot find the answer, say that you do not know.
"""

JUDGE_PROMPT = """
You are grading answers to questions about academic papers.
You will be given a question, a reference answer, and a candidate answer.
The candidate is CORRECT if it contains the key facts of the reference answer
and does not contradict it. Extra detail is fine; missing or wrong key facts
are not. Reply with exactly one word: CORRECT or INCORRECT.
"""

CHUNK_SIZE = 1500    # characters per chunk (~370 tokens; EmbeddingGemma accepts up to 2048)
CHUNK_OVERLAP = 200  # characters shared between neighboring chunks
EMBED_BATCH = 64     # chunks per embeddings request, to stay under the service's rate limit
MAX_TURNS = 8        # stop the agent if it keeps searching without answering

SOPHIA_BASE_URL = 'https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1'

# EmbeddingGemma was trained with different prefixes for queries and documents
# (see the model card). vLLM does not add them, so we do.
QUERY_PREFIX = 'task: search result | query: '
DOCUMENT_PREFIX = 'title: none | text: '


@dataclass
class SearchContext:
    '''State shared with the search_papers tool during one question. The
    Agents SDK passes this object to the tool through RunContextWrapper; the
    model never sees it.'''
    collection: chromadb.Collection
    embed_client: OpenAI
    embed_model: str
    # Every paper returned by search_papers while answering this question.
    retrieved_papers: set[str] = field(default_factory=set)


def read_paper(path: Path) -> str:
    '''Returns the text of a .pdf or .txt paper.'''
    if path.suffix.lower() == '.pdf':
        return '\n'.join(page.extract_text() or '' for page in PdfReader(path).pages)
    return path.read_text()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    '''Splits text into overlapping fixed-size windows. The overlap keeps a
    sentence that straddles a boundary intact in at least one chunk.'''
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def embed(client: OpenAI, model: str, texts: list[str], prefix: str) -> list[list[float]]:
    '''Embeds texts with the remote embedding model, in batches.'''
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [prefix + t for t in texts[i:i + EMBED_BATCH]]
        response = client.embeddings.create(model=model, input=batch, encoding_format='float')
        vectors.extend(item.embedding for item in response.data)
    return vectors


def ingest(papers: dict[str, str], embed_client: OpenAI, embed_model: str) -> chromadb.Collection:
    '''Chunks every paper, embeds the chunks, and stores them in a fresh
    in-memory Chroma collection. embedding_function=None tells Chroma we will
    supply the vectors ourselves instead of using its built-in model.'''

    # TODO: Ingest papers into ChromaDB


@function_tool
def search_papers(ctx: RunContextWrapper[SearchContext], query: str, k: int = 5) -> str:
    '''Searches the paper collection for passages relevant to the query and
    returns the k best matches, each labeled with its source paper. Phrase
    the query as a description of the information you need.

    Args:
        query: What to search for.
        k: Number of passages to return (default 5).
    '''
    
    # TODO: Write a search tool to query for passages to append to the context


def ask(agent: Agent[SearchContext], context: SearchContext, question: str) -> tuple[str, int, int]:
    '''Runs the agent on one question. The Runner handles the tool-calling
    loop: call the model, run any requested tools, feed the results back, and
    repeat until the model answers or MAX_TURNS is reached.
    Returns the final answer, the number of searches, and the total input
    tokens summed over every model call.'''
    try:
        result = Runner.run_sync(agent, question, context=context, max_turns=MAX_TURNS)
    except MaxTurnsExceeded:
        return 'ERROR: no answer after the maximum number of turns', 0, 0

    searches = [item for item in result.new_items if isinstance(item, ToolCallItem)]
    for item in searches:
        print(f'    search: {item.raw_item.arguments}')
    return str(result.final_output), len(searches), result.context_wrapper.usage.input_tokens


def judge(agent: Agent, question: str, reference: str, candidate: str) -> bool:
    '''Asks the judge agent whether the candidate answer matches the reference.'''
    result = Runner.run_sync(
        agent,
        f'Question: {question}\n\nReference answer: {reference}\n\nCandidate answer: {candidate}',
    )
    verdict = str(result.final_output).strip().upper()
    # Check INCORRECT first, since "CORRECT" is a substring of it.
    return 'INCORRECT' not in verdict and 'CORRECT' in verdict


def select_corpus(paths: list[Path], sources: set[str], num_papers: int) -> list[Path]:
    '''Picks the papers to ingest: every question's source paper, filled out
    with other papers in sorted order up to num_papers.'''
    if len(sources) > num_papers:
        raise SystemExit(f'--num-papers must be at least {len(sources)} to include every source paper')
    others = [p for p in paths if p.name not in sources]
    return [p for p in paths if p.name in sources] + others[:num_papers - len(sources)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--papers', type=Path, required=True, help='Folder of .pdf/.txt papers')
    parser.add_argument('--questions', type=Path, required=True,
                        help='JSON list of {question, answer, category, source_paper}')
    parser.add_argument('-m', '--model', default='inkling-bf16', help='Model under test')
    parser.add_argument('-j', '--judge-model', default='beam-1.0-bf16', help='Model used to grade answers')
    parser.add_argument('-e', '--embedding-model', default='google/embeddinggemma-300m', help='Embedding model')
    parser.add_argument('--embedding-base-url', default=SOPHIA_BASE_URL,
                        help='OpenAI-compatible endpoint serving the embedding model (default: Sophia)')
    parser.add_argument('-n', '--num-papers', type=int, required=True, help='Number of papers to ingest')
    args = parser.parse_args()

    # The SDK uploads traces to OpenAI by default, which fails with an ALCF token.
    set_tracing_disabled(True)

    # Chat/judge client reads OPENAI_BASE_URL; the embedding client is pinned to
    # its own cluster. Both use the same ALCF token from OPENAI_API_KEY.
    chat_client = AsyncOpenAI()
    embed_client = OpenAI(base_url=args.embedding_base_url)

    # OpenAIResponsesModel uses the Responses API, the same one 01_coding_agent.py
    # calls directly. Passing our own client points it at OPENAI_BASE_URL.
    answer_agent = Agent[SearchContext](
        name='paper-qa',
        instructions=ANSWER_PROMPT,
        model=OpenAIResponsesModel(model=args.model, openai_client=chat_client),
        tools=[search_papers],
    )
    judge_agent = Agent(
        name='judge',
        instructions=JUDGE_PROMPT,
        model=OpenAIResponsesModel(model=args.judge_model, openai_client=chat_client),
    )

    questions = json.loads(args.questions.read_text())

    paths = sorted(p for p in args.papers.iterdir() if p.suffix.lower() in ('.pdf', '.txt'))
    if not 1 <= args.num_papers <= len(paths):
        raise SystemExit(f'--num-papers must be between 1 and {len(paths)} (papers found in {args.papers})')
    sources = {q['source_paper'] for q in questions}
    missing = sources - {p.name for p in paths}
    if missing:
        raise SystemExit(f'source_paper not found in {args.papers}: {sorted(missing)}')

    corpus = select_corpus(paths, sources, args.num_papers)
    collection = ingest({p.name: read_paper(p) for p in corpus}, embed_client, args.embedding_model)
    print(f'Ingested {len(corpus)} papers as {collection.count()} chunks')

    # category -> [num_correct, num_retrieval_hits, num_total]
    scores: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for i, q in enumerate(questions, 1):
        # A fresh context per question, so retrieved_papers starts empty.
        context = SearchContext(collection, embed_client, args.embedding_model)
        print(f'\n[{i}/{len(questions)}] ({q["category"]}) Q: {q["question"]}')
        try:
            answer, searches, tokens = ask(answer_agent, context, q['question'])
            correct = judge(judge_agent, q['question'], q['answer'], answer)
        except Exception as e:
            answer, searches, tokens, correct = f'ERROR: {e}', 0, 0, False

        # Did any search return a chunk from the paper that holds the answer?
        hit = q['source_paper'] in context.retrieved_papers
        scores[q['category']][0] += correct
        scores[q['category']][1] += hit
        scores[q['category']][2] += 1
        print(f'  {"CORRECT" if correct else "INCORRECT"} | searches: {searches} | '
              f'source retrieved: {"yes" if hit else "no"} | input tokens: {tokens:,}')
        print(f'  A: {answer.strip()}')

    print('\n=== Results by category ===')
    print(f'{"":>12}  {"correct":>10}  {"retrieved":>10}')
    for category, (num_correct, hits, total) in sorted(scores.items()):
        print(f'{category:>12}  {num_correct:>4}/{total:<5}  {hits:>4}/{total:<5}')
    totals = [sum(s[i] for s in scores.values()) for i in range(3)]
    print(f'{"overall":>12}  {totals[0]:>4}/{totals[2]:<5}  {totals[1]:>4}/{totals[2]:<5}')


if __name__ == '__main__':
    main()
