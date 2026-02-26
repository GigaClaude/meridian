"""Shared fixtures for Meridian tests.

All integration tests require running Qdrant (port 6333) and Ollama (port 11434).
"""

import pytest
import pytest_asyncio

from meridian.config import config
from meridian.gateway import MemoryGateway
from meridian.storage import StorageLayer
from meridian.workers import WorkerPool

TEST_COLLECTION = "test_meridian_pytest"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def storage():
    """StorageLayer pointed at an isolated test collection."""
    original = config.qdrant_collection
    config.qdrant_collection = TEST_COLLECTION
    s = StorageLayer()
    await s.init_db()
    yield s
    try:
        s.qdrant.delete_collection(TEST_COLLECTION)
    except Exception:
        pass
    config.qdrant_collection = original
    await s.close()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def gateway():
    gw = MemoryGateway()
    yield gw
    await gw.close()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def workers():
    wp = WorkerPool()
    yield wp
    await wp.close()
