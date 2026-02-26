"""Pytest-compatible recall quality tests.

Wraps the recall quality test suite for `pytest` discovery.
Requires running Qdrant + Ollama services.

    pytest tests/test_recall_pytest.py -v -m integration
"""

import pytest
import pytest_asyncio

from qdrant_client.models import Distance, VectorParams

from .test_recall_quality import FIXTURES, TEST_QUERIES

COLLECTION = "test_recall_pytest"


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def seeded_collection(storage, workers):
    """Seed fixture memories into an isolated test collection."""
    from meridian.config import config

    original = config.qdrant_collection
    config.qdrant_collection = COLLECTION

    # Recreate collection
    try:
        storage.qdrant.delete_collection(COLLECTION)
    except Exception:
        pass

    dim = storage._probe_embed_dim()
    storage.qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    for fix in FIXTURES:
        await storage.remember(
            {
                "content": fix["content"],
                "type": fix["type"],
                "tags": fix["tags"],
                "importance": fix["importance"],
                "source": fix.get("source", "test"),
                "related_to": [],
            },
            workers,
        )

    yield storage

    # Teardown
    try:
        storage.qdrant.delete_collection(COLLECTION)
    except Exception:
        pass
    config.qdrant_collection = original


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "test_case",
    TEST_QUERIES,
    ids=[t["description"] for t in TEST_QUERIES],
)
async def test_recall_quality(test_case, seeded_collection, gateway):
    """Each query should return expected fragments and citations."""
    result = await gateway.recall(
        test_case["query"],
        test_case["scope"],
        test_case["max_tokens"],
        seeded_collection,
    )

    summary = result["results"]

    # Check fragments
    for frag in test_case["expect_fragments"]:
        assert frag.lower() in summary.lower(), (
            f"Missing fragment '{frag}' in response for: {test_case['description']}\n"
            f"Response: {summary[:300]}"
        )

    # Check citations
    citation_count = summary.count("[mem_")
    assert citation_count >= test_case["min_citations"], (
        f"Expected >= {test_case['min_citations']} citations, got {citation_count}\n"
        f"Response: {summary[:300]}"
    )
