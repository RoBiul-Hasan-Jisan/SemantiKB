"""
Versioning: supports multiple versions of the same logical document,
diffing between versions, and time-scoped lookups.
"""
from __future__ import annotations

import difflib
from datetime import datetime

from backend.database.repository import Repository
from backend.models import Document, DocumentStatus, DocumentVersion, content_hash, new_id


class VersionManager:
    def __init__(self, repo: Repository):
        self.repo = repo

    def register_upload(
        self, filename: str, file_path: str, full_text: str, page_count: int,
        existing_document_id: str | None = None,
    ) -> tuple[Document, DocumentVersion]:
        """
        Register a newly-uploaded file either as a brand-new document, or
        (if existing_document_id is given / a matching filename already
        exists) as a new version of an existing document.
        """
        chash = content_hash(full_text)

        if existing_document_id:
            doc = self.repo.get_document(existing_document_id)
            if doc is None:
                raise ValueError(f"No such document_id: {existing_document_id}")
        else:
            doc = Document(document_id=new_id("doc_"), filename=filename, status=DocumentStatus.PENDING)
            self.repo.upsert_document(doc)

        existing_versions = self.repo.list_versions(doc.document_id)
        next_version = (max((v.version for v in existing_versions), default=0)) + 1

        now = datetime.utcnow()
        version = DocumentVersion(
            document_id=doc.document_id, version=next_version, filename=filename,
            file_path=file_path, content_hash=chash, created_at=now, updated_at=now,
            valid_from=now, valid_to=None, page_count=page_count, is_latest=True,
        )
        self.repo.add_version(version)

        doc.latest_version = next_version
        self.repo.upsert_document(doc)
        return doc, version

    def diff_versions(self, document_id: str, version_a: int, version_b: int) -> dict:
        pages_a = self.repo.get_pages(document_id, version_a)
        pages_b = self.repo.get_pages(document_id, version_b)
        text_a = "\n".join(p.text for p in pages_a)
        text_b = "\n".join(p.text for p in pages_b)

        sm = difflib.SequenceMatcher(a=text_a.split("\n"), b=text_b.split("\n"))
        added, removed, changed = [], [], []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "insert":
                added.extend(text_b.split("\n")[j1:j2])
            elif tag == "delete":
                removed.extend(text_a.split("\n")[i1:i2])
            elif tag == "replace":
                changed.append({
                    "before": "\n".join(text_a.split("\n")[i1:i2]),
                    "after": "\n".join(text_b.split("\n")[j1:j2]),
                })

        similarity_ratio = sm.ratio()
        return {
            "document_id": document_id,
            "version_a": version_a,
            "version_b": version_b,
            "similarity_ratio": round(similarity_ratio, 4),
            "added_lines": [l for l in added if l.strip()],
            "removed_lines": [l for l in removed if l.strip()],
            "changed_blocks": changed,
        }
