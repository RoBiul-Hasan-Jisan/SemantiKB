import tempfile
from pathlib import Path

import pytest

from backend.database.repository import Repository
from backend.models import Page
from backend.versioning.version_manager import VersionManager


@pytest.fixture()
def repo(tmp_path):
    db_path = tmp_path / "test.db"
    return Repository(db_path)


def test_register_upload_creates_new_document(repo):
    vm = VersionManager(repo)
    doc, version = vm.register_upload("policy.txt", "/tmp/policy.txt", "Some policy text", page_count=1)
    assert version.version == 1
    assert doc.latest_version == 1


def test_register_upload_second_time_creates_new_version(repo):
    vm = VersionManager(repo)
    doc, v1 = vm.register_upload("policy.txt", "/tmp/policy_v1.txt", "Old policy text", page_count=1)
    doc2, v2 = vm.register_upload(
        "policy.txt", "/tmp/policy_v2.txt", "New policy text", page_count=1,
        existing_document_id=doc.document_id,
    )
    assert doc2.document_id == doc.document_id
    assert v2.version == 2
    latest = repo.get_latest_version(doc.document_id)
    assert latest.version == 2

    old = repo.get_version(doc.document_id, 1)
    assert old.is_latest is False
    assert old.valid_to is not None


def test_diff_versions_detects_changes(repo):
    vm = VersionManager(repo)
    doc, v1 = vm.register_upload("policy.txt", "/tmp/p1.txt", "Line A\nLine B\nLine C", page_count=1)
    repo.add_pages([Page(document_id=doc.document_id, version=1, page_number=1, text="Line A\nLine B\nLine C")])

    doc2, v2 = vm.register_upload(
        "policy.txt", "/tmp/p2.txt", "Line A\nLine B changed\nLine C\nLine D",
        page_count=1, existing_document_id=doc.document_id,
    )
    repo.add_pages([Page(document_id=doc.document_id, version=2, page_number=1,
                          text="Line A\nLine B changed\nLine C\nLine D")])

    diff = vm.diff_versions(doc.document_id, 1, 2)
    assert diff["version_a"] == 1
    assert diff["version_b"] == 2
    assert any("Line D" in l for l in diff["added_lines"])
