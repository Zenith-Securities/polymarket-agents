import json
import sys
from enum import Enum
from pathlib import Path

from typing import Iterable, Optional

import typer
from devtools import pprint

# Ensure project root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.polymarket.polymarket import Polymarket
from agents.connectors.chroma import PolymarketRAG
from agents.connectors.news import News
from agents.application.trade import Trader
from agents.application.executor import Executor
from agents.application.creator import Creator

app = typer.Typer()
polymarket = Polymarket()
newsapi_client = News()
polymarket_rag = PolymarketRAG()


class RagSortOption(str, Enum):
    SCORE = "score"
    VOLUME = "volume"


@app.command()
def get_all_markets(limit: int = 5, sort_by: str = "spread") -> None:
    """
    Query Polymarket's markets
    """
    print(f"limit: int = {limit}, sort_by: str = {sort_by}")
    markets = polymarket.get_all_markets()
    markets = polymarket.filter_markets_for_trading(markets)
    if sort_by == "spread":
        markets = sorted(markets, key=lambda x: x.spread, reverse=True)
    markets = markets[:limit]
    pprint(markets)


@app.command()
def get_relevant_news(keywords: str) -> None:
    """
    Use NewsAPI to query the internet
    """
    articles = newsapi_client.get_articles_for_cli_keywords(keywords)
    pprint(articles)


@app.command()
def get_all_events(limit: int = 5, sort_by: str = "number_of_markets") -> None:
    """
    Query Polymarket's events
    """
    print(f"limit: int = {limit}, sort_by: str = {sort_by}")
    events = polymarket.get_all_events()
    events = polymarket.filter_events_for_trading(events)
    if sort_by == "number_of_markets":
        events = sorted(events, key=lambda x: len(x.markets), reverse=True)
    events = events[:limit]
    pprint(events)


@app.command()
def create_local_markets_rag(local_directory: str) -> None:
    """
    Create a local markets database for RAG
    """
    polymarket_rag.create_local_markets_rag(local_directory=local_directory)


@app.command()
def query_local_markets_rag(
    vector_db_directory: str,
    query: str,
    k: int = typer.Option(
        4, min=1, help="Number of results to return from the vector store."
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Optional path to write the query results as JSON.",
    ),
    sort_by: RagSortOption = typer.Option(
        RagSortOption.SCORE,
        "--sort-by",
        "-s",
        case_sensitive=False,
        help="Sort the retrieved results. Use 'volume' to sort by highest market volume.",
    ),
) -> None:
    """
    RAG over a local database of Polymarket's events
    """
    response = polymarket_rag.query_local_markets_rag(
        local_directory=vector_db_directory, query=query, k=k
    )

    def _get_target_indices(
        documents: Iterable
    ) -> dict[Path, list[tuple[int, dict]]]:
        grouped: dict[Path, list[tuple[int, dict]]] = {}
        for doc, _score in documents:
            try:
                existing_volume = float(doc.metadata.get("volume"))
                if existing_volume:
                    continue
            except (TypeError, ValueError):
                pass

            source_path = doc.metadata.get("source")
            seq_num = doc.metadata.get("seq_num")
            if not source_path or seq_num is None:
                doc.metadata["volume"] = 0.0
                continue

            try:
                index = int(seq_num)
            except (TypeError, ValueError):
                doc.metadata["volume"] = 0.0
                continue

            grouped.setdefault(Path(source_path), []).append((index, doc.metadata))
        return grouped

    def _extract_market_info(market: dict) -> dict[str, object]:
        info: dict[str, object] = {}

        for key in ("volumeNum", "volume", "volumeUSD", "volumeUsd"):
            raw_value = market.get(key)
            if raw_value is None:
                continue
            try:
                info["volume"] = float(raw_value)
                break
            except (TypeError, ValueError):
                continue
        if "volume" not in info:
            info["volume"] = 0.0

        market_slug = market.get("slug")
        if isinstance(market_slug, str) and market_slug:
            info["market_slug"] = market_slug

        events = market.get("events")
        if isinstance(events, list):
            event_slugs = [
                event.get("slug")
                for event in events
                if isinstance(event, dict) and isinstance(event.get("slug"), str)
            ]
            event_slugs = [slug for slug in event_slugs if slug]
            if event_slugs:
                info["event_slugs"] = event_slugs
                info["event_slug"] = event_slugs[0]

        return info

    def _load_volumes_for_indices(path: Path, indices: list[int]) -> dict[int, dict[str, object]]:
        requested = sorted(set(i for i in indices if i >= 0))
        if not requested:
            return {}

        decoder = json.JSONDecoder()
        results: dict[int, dict[str, object]] = {
            i: {"volume": 0.0} for i in requested
        }
        max_index = requested[-1]
        current_target_iter = iter(requested)
        try:
            current_target = next(current_target_iter)
        except StopIteration:
            return results

        buffer = ""
        index = -1
        array_started = False

        try:
            with path.open("r") as handle:
                while True:
                    if not buffer:
                        chunk = handle.read(4096)
                        if not chunk:
                            break
                        buffer += chunk

                    buffer = buffer.lstrip()
                    if not array_started:
                        if not buffer:
                            continue
                        if buffer[0] == "[":
                            buffer = buffer[1:]
                            array_started = True
                        else:
                            raise ValueError("Unexpected JSON format: expected array")
                        continue

                    if not buffer:
                        continue

                    if buffer[0] == ",":
                        buffer = buffer[1:]
                        continue
                    if buffer[0] == "]":
                        break

                    try:
                        market, offset = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        # Need more data from file
                        chunk = handle.read(4096)
                        if not chunk:
                            break
                        buffer += chunk
                        continue

                    buffer = buffer[offset:]
                    index += 1

                    if index > max_index:
                        break

                    while index >= current_target:
                        if index == current_target and isinstance(market, dict):
                            results[current_target] = _extract_market_info(market)
                        try:
                            current_target = next(current_target_iter)
                        except StopIteration:
                            return results
        except (OSError, ValueError):
            return results

        return results

    grouped_indices = _get_target_indices(response)
    for source_path, selector in grouped_indices.items():
        market_info = _load_volumes_for_indices(source_path, [idx for idx, _ in selector])
        for idx, metadata in selector:
            info = market_info.get(idx, {})
            for key, value in info.items():
                metadata[key] = value

    if sort_by == RagSortOption.VOLUME:
        response = sorted(
            response, key=lambda item: item[0].metadata.get("volume", 0.0), reverse=True
        )
    pprint(response)
    if output_file:
        serialized_response = [
            {
                "metadata": doc.metadata,
                "page_content": doc.page_content,
                "score": score,
            }
            for doc, score in response
        ]
        output_file = output_file.expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(serialized_response, indent=2))


@app.command()
def ask_superforecaster(event_title: str, market_question: str, outcome: str) -> None:
    """
    Ask a superforecaster about a trade
    """
    print(
        f"event: str = {event_title}, question: str = {market_question}, outcome (usually yes or no): str = {outcome}"
    )
    executor = Executor()
    response = executor.get_superforecast(
        event_title=event_title, market_question=market_question, outcome=outcome
    )
    print(f"Response:{response}")


@app.command()
def create_market() -> None:
    """
    Format a request to create a market on Polymarket
    """
    c = Creator()
    market_description = c.one_best_market()
    print(f"market_description: str = {market_description}")


@app.command()
def ask_llm(user_input: str) -> None:
    """
    Ask a question to the LLM and get a response.
    """
    executor = Executor()
    response = executor.get_llm_response(user_input)
    print(f"LLM Response: {response}")


@app.command()
def ask_polymarket_llm(user_input: str) -> None:
    """
    What types of markets do you want trade?
    """
    executor = Executor()
    response = executor.get_polymarket_llm(user_input=user_input)
    print(f"LLM + current markets&events response: {response}")


@app.command()
def run_autonomous_trader() -> None:
    """
    Let an autonomous system trade for you.
    """
    trader = Trader()
    trader.one_best_trade()


if __name__ == "__main__":
    app()
