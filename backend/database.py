"""Local persistence: SQLite by default, PostgreSQL (JSONB + GIN) when NEPDOC_DATABASE_URL is set.

Tables
  documents   one row per upload: status, metadata, the full JSON record (JSON/JSONB)
  components  one row per detected component, for SQL queries such as
              "every stamp", "documents with Darta No. X", "text containing …"
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine,
                        delete, event, func, or_, select)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import Settings, get_settings

JsonType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    filename: Mapped[str] = mapped_column(String(200))           # display only
    content_type: Mapped[str] = mapped_column(String(40))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued|processing|done|failed
    stage: Mapped[str | None] = mapped_column(String(200), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    options: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    record: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    full_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # normalised, for search
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    components: Mapped[list["Component"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Component(Base):
    __tablename__ = "components"
    __table_args__ = (Index("ix_components_doc_order", "document_id", "reading_order"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    component_id: Mapped[str] = mapped_column(String(16))        # e.g. p1-c003
    page: Mapped[int] = mapped_column(Integer)
    reading_order: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(24), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    x0: Mapped[int] = mapped_column(Integer)
    y0: Mapped[int] = mapped_column(Integer)
    x1: Mapped[int] = mapped_column(Integer)
    y1: Mapped[int] = mapped_column(Integer)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_normalized: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    script: Mapped[str | None] = mapped_column(String(8), nullable=True)
    asset_path: Mapped[str | None] = mapped_column(String(200), nullable=True)

    document: Mapped[Document] = relationship(back_populates="components")


class Database:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        url = self.settings.db_url
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _pragmas(conn, _):
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")     # API reads while the worker writes
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA secure_delete=ON")     # overwrite deleted PII pages
                cur.close()
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def init(self) -> None:
        Base.metadata.create_all(self.engine)
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as conn:
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_documents_record_gin ON documents USING GIN (record jsonb_path_ops)")

    # ---- writes --------------------------------------------------------------------
    def create(self, *, filename: str, content_type: str, sha256: str, size_bytes: int, options: dict) -> Document:
        with self.Session.begin() as s:
            doc = Document(filename=filename, content_type=content_type, sha256=sha256,
                           size_bytes=size_bytes, options=options)
            s.add(doc)
        return doc

    def set_stage(self, doc_id: str, stage: str, status: str = "processing") -> None:
        with self.Session.begin() as s:
            doc = s.get(Document, doc_id)
            if doc:
                doc.status, doc.stage = status, stage[:200]

    def save_result(self, doc_id: str, record: dict) -> None:
        with self.Session.begin() as s:
            doc = s.get(Document, doc_id)
            if doc is None:
                return
            doc.record, doc.status, doc.stage = record, "done", None
            doc.page_count = record.get("page_count")
            doc.full_text = " ".join(c["text_normalized"] for c in record["components"] if c.get("text_normalized"))
            doc.finished_at = utcnow()
            s.execute(delete(Component).where(Component.document_id == doc_id))
            for c in record["components"]:
                b = c["bbox"]
                s.add(Component(
                    document_id=doc_id, component_id=c["id"], page=c["page"], reading_order=c["reading_order"],
                    type=c["type"], confidence=c["confidence"], x0=b["x0"], y0=b["y0"], x1=b["x1"], y1=b["y1"],
                    text=c.get("text"), text_normalized=c.get("text_normalized"),
                    text_confidence=c.get("text_confidence"), script=c.get("script"),
                    asset_path=(c.get("asset") or {}).get("png")))

    def fail(self, doc_id: str, error: str) -> None:
        with self.Session.begin() as s:
            doc = s.get(Document, doc_id)
            if doc:
                doc.status, doc.error, doc.stage, doc.finished_at = "failed", error[:1000], None, utcnow()

    def delete(self, doc_id: str) -> bool:
        with self.Session.begin() as s:
            doc = s.get(Document, doc_id)
            if doc is None:
                return False
            s.delete(doc)
        for folder in (self.settings.assets_dir / doc_id, self.settings.pages_dir / doc_id):
            shutil.rmtree(folder, ignore_errors=True)
        return True

    def purge_older_than(self, days: int) -> int:
        if days <= 0:
            return 0
        cutoff = utcnow() - timedelta(days=days)
        with self.Session() as s:
            ids = list(s.scalars(select(Document.id).where(Document.created_at < cutoff)))
        return sum(self.delete(i) for i in ids)

    def requeue_interrupted(self) -> list[str]:
        """Jobs that were mid-flight when the app stopped."""
        with self.Session.begin() as s:
            docs = list(s.scalars(select(Document).where(Document.status.in_(("queued", "processing")))))
            for d in docs:
                d.status, d.error, d.stage = "failed", "Interrupted by a restart — please upload again", None
            return [d.id for d in docs]

    # ---- reads ---------------------------------------------------------------------
    def get(self, doc_id: str) -> Document | None:
        with self.Session() as s:
            return s.get(Document, doc_id)

    def list(self, q: str | None = None, limit: int = 50, offset: int = 0) -> list[Document]:
        with self.Session() as s:
            stmt = select(Document).order_by(Document.created_at.desc())
            if q:
                like = f"%{q.strip().lower()}%"
                stmt = stmt.where(or_(func.lower(Document.full_text).like(like),
                                      func.lower(Document.filename).like(like)))
            return list(s.scalars(stmt.limit(limit).offset(offset)))

    def find_components(self, type: str | None = None, text: str | None = None, limit: int = 200) -> list[Component]:
        with self.Session() as s:
            stmt = select(Component).order_by(Component.document_id, Component.reading_order)
            if type:
                stmt = stmt.where(Component.type == type)
            if text:
                stmt = stmt.where(Component.text_normalized.like(f"%{text.strip().lower()}%"))
            return list(s.scalars(stmt.limit(limit)))

    def queue_length(self) -> int:
        with self.Session() as s:
            return s.scalar(select(func.count()).select_from(Document)
                            .where(Document.status.in_(("queued", "processing")))) or 0


def asset_root(settings: Settings) -> Path:
    return settings.data_dir
