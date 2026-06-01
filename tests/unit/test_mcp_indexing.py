"""Tests for MCP index_repo indexing behavior."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from clean.db.metadata import MetadataStore
from clean.db.models import ProjectRecord
from clean.local.mcp_server import _handle_index_local_path, _handle_index_repo
from clean.mcp.shared import _make_project_id


class _FakeStore:
    def __init__(self) -> None:
        self.cleared: list[str] = []

    def clear(self, project_id: str) -> None:
        self.cleared.append(project_id)


class _FakeIndexer:
    def __init__(self) -> None:
        self.calls = 0

    def index(self, path: str, project_id: str | None = None) -> dict:
        self.calls += 1
        return {
            "status": "success",
            "functions_indexed": 1,
            "files_processed": 1,
        }


class _FakeContainer:
    def __init__(self) -> None:
        self.indexer = _FakeIndexer()
        self.store = _FakeStore()


class _SlowRepoManager:
    def __init__(self, local_path: str) -> None:
        self.local_path = local_path
        self.clone_started = threading.Event()
        self.delete_calls = 0

    def repo_path(self, repo: str, branch: str | None = None) -> str:
        return self.local_path

    def exists(self, repo: str, branch: str | None = None) -> bool:
        return False

    def delete(self, repo: str, branch: str | None = None) -> None:
        self.delete_calls += 1

    def clone(
        self, clone_url: str, repo: str, branch: str | None = None
    ) -> str:
        self.clone_started.set()
        time.sleep(0.2)
        return self.local_path


@pytest.mark.asyncio
async def test_index_local_path_defaults_to_background(tmp_path):
    metadata = MetadataStore(str(tmp_path / "metadata.db"))
    container = _FakeContainer()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "example.py").write_text("def hello():\n    return 'world'\n")

    response = await _handle_index_local_path(
        {"path": str(project_dir)}, container, metadata
    )

    assert "Started indexing local/project in the background" in response[0].text

    project_id = _make_project_id("local/project", None)
    for _ in range(20):
        project = metadata.get_project(project_id)
        if project and project.status == "ready":
            break
        await asyncio.sleep(0.05)

    project = metadata.get_project(project_id)
    assert project is not None
    assert project.status == "ready"
    assert project.entity_count == 1
    assert container.indexer.calls == 1


@pytest.mark.asyncio
async def test_index_github_repo_background_registers_before_clone_finishes(tmp_path):
    metadata = MetadataStore(str(tmp_path / "metadata.db"))
    container = _FakeContainer()
    repo_dir = tmp_path / "repos" / "owner" / "project"
    repo_dir.mkdir(parents=True)
    (repo_dir / "example.py").write_text("def hello():\n    return 'world'\n")
    repo_manager = _SlowRepoManager(str(repo_dir))

    response = await _handle_index_repo(
        {"repo": "owner/project"}, container, metadata, repo_manager
    )

    assert (
        "Started cloning and indexing owner/project in the background"
        in response[0].text
    )
    for _ in range(20):
        if repo_manager.clone_started.is_set():
            break
        await asyncio.sleep(0.05)
    assert repo_manager.clone_started.is_set()

    project_id = _make_project_id("owner/project", None)
    project = metadata.get_project(project_id)
    assert project is not None
    assert project.status in {"cloning", "indexing", "ready"}

    for _ in range(20):
        project = metadata.get_project(project_id)
        if project and project.status == "ready":
            break
        await asyncio.sleep(0.05)

    project = metadata.get_project(project_id)
    assert project is not None
    assert project.status == "ready"
    assert project.entity_count == 1
    assert project.local_path == str(repo_dir)
    assert container.indexer.calls == 1


@pytest.mark.asyncio
async def test_index_local_path_replaces_stale_indexing_record(tmp_path):
    metadata = MetadataStore(str(tmp_path / "metadata.db"))
    container = _FakeContainer()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "example.py").write_text("def hello():\n    return 'world'\n")

    project_id = _make_project_id("local/project", None)
    metadata.save_project(
        ProjectRecord(
            project_id=project_id,
            repo_full_name="local/project",
            branch=None,
            local_path=str(project_dir),
            status="indexing",
            created_at=(datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
        )
    )

    response = await _handle_index_local_path(
        {"path": str(project_dir), "background": False}, container, metadata
    )

    assert "Indexed local/project" in response[0].text
    assert container.store.cleared == [project_id]
    project = metadata.get_project(project_id)
    assert project is not None
    assert project.status == "ready"
