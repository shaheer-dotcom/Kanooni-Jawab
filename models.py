"""
ingestion/models.py — Per-Jurisdiction Postgres Table Definitions
=================================================================
One table per (jurisdiction × document_type) combination.

Tables created:
  Cases:       hk_case_metadata | pak_case_metadata | uk_case_metadata
               ca_case_metadata | us_case_metadata  | au_case_metadata
               in_case_metadata

  Legislation: hk_legis_metadata | pak_legis_metadata | uk_legis_metadata
               ca_legis_metadata  | us_legis_metadata  | au_legis_metadata
               in_legis_metadata

Qdrant uses one collection per jurisdiction (e.g. `canada`). Cases and
legislation share that collection and are split by the `doc_type` payload
field. Isolation across jurisdictions is the collection name plus the
`jurisdiction` payload filter.

Usage:
    from ingestion.models import CASE_MODEL_REGISTRY, LEGIS_MODEL_REGISTRY, init_db

    ModelClass = CASE_MODEL_REGISTRY["pakistan"]   # → PakCaseMetadata
    LegisClass = LEGIS_MODEL_REGISTRY["hong_kong"] # → HKLegisMetadata
"""

from __future__ import annotations

from typing import Type

from sqlalchemy import BigInteger, Column, Text, create_engine as sa_create_engine, text
from sqlalchemy.engine.url import make_url
from sqlmodel import Field, SQLModel, create_engine

from config import Config


# ── Engine ────────────────────────────────────────────────────────────────────
engine = create_engine(Config.DATABASE_URL, echo=False)


# =============================================================================
# BASE MODELS  (table=False — never instantiated directly)
# =============================================================================

class BaseCaseMetadata(SQLModel, table=False):
    """
    Shared schema for every jurisdiction's case table.
    Subclasses add table=True + __tablename__ + any jurisdiction-specific columns.
    """
    id:             str = Field(primary_key=True)
    file_name:      str = Field(default="", index=True)
    file_hash:      str = Field(default="", index=True)

    # Identity
    case_name:        str = Field(default="")
    jurisdiction:     str = Field(default="", index=True)   # canonical e.g. "Pakistan"
    jurisdiction_key: str = Field(default="", index=True)   # registry key e.g. "pakistan"

    # Structured metadata
    case_number:       str = Field(default="")
    case_citation:     str = Field(default="", index=True)
    case_court:        str = Field(default="", index=True)
    case_country:      str = Field(default="")
    case_type:         str = Field(default="")               # civil/criminal/constitutional/…
    case_decision_date: str = Field(default="")
    case_year:         int = Field(default=0, index=True)

    # Parties
    case_petitioner_name: str = Field(default="")
    case_petitioner_role: str = Field(default="")
    case_respondent_name: str = Field(default="")
    case_respondent_role: str = Field(default="")

    # Substance
    case_submission_dates:     str = Field(default="", sa_type=Text)
    case_summary:              str = Field(default="", sa_type=Text)
    case_legal_references:     str = Field(default="", sa_type=Text)
    case_successful_party:     str = Field(default="")
    case_costs_order:          str = Field(default="")
    case_reasons_for_deviation: str = Field(default="", sa_type=Text)
    case_key_issues:           str = Field(default="", sa_type=Text)
    case_key_points:           str = Field(default="", sa_type=Text)
    case_keywords:             str = Field(default="", sa_type=Text)
    case_key_phrases:          str = Field(default="", sa_type=Text)

    # Storage / audit
    file_url:      str | None = Field(default=None)
    ingest_status: str        = Field(default="success")   # success | failed
    ingest_error:  str | None = Field(default=None)


class BaseLegisMetadata(SQLModel, table=False):
    """Shared schema for every jurisdiction's legislation table."""
    # Use sa_type (not sa_column=Column(...)): a shared Column instance cannot be
    # attached to multiple jurisdiction tables (HK, Pak, UK, …).
    id: int | None = Field(default=None, primary_key=True, sa_type=BigInteger)
    file_name:      str = Field(default="", index=True)
    file_hash:      str = Field(default="", index=True)

    # Identity
    legislation_name: str = Field(default="")
    jurisdiction:     str = Field(default="", index=True)
    jurisdiction_key: str = Field(default="", index=True)

    # Normalized universal legislation identifier
    # HK  → cap number e.g. "486"
    # Pak → "5-OF-1997"
    # UK  → "2006"  (year of Act)
    # CA  → "RSC-1985-C-C-46"
    act_number: str = Field(default="", index=True)
    act_label:  str = Field(default="")             # "Cap 486" / "Act No. 5 of 1997"

    # Raw as-extracted strings (kept for dedup + backward compat)
    legislation_number:      str = Field(default="")
    full_legislation_number: str = Field(default="")
    legislation_version:     str = Field(default="")
    legislation_effective_date: str = Field(default="", index=True)

    # Substance
    legislation_type:            str = Field(default="")   # primary/subsidiary/regulation
    legislation_summary:         str = Field(default="", sa_type=Text)
    legislation_country:         str = Field(default="")
    legislation_legal_references: str = Field(default="", sa_type=Text)
    legislation_key_sections:    str = Field(default="", sa_type=Text)
    legislation_key_points:      str = Field(default="", sa_type=Text)
    legislation_keywords:        str = Field(default="", sa_type=Text)
    legislation_key_phrases:     str = Field(default="", sa_type=Text)

    # Storage / audit
    file_url:      str | None = Field(default=None)
    ingest_status: str        = Field(default="success")
    ingest_error:  str | None = Field(default=None)


# =============================================================================
# HONG KONG  →  hk_case_metadata  |  hk_legis_metadata
# =============================================================================

class HKCaseMetadata(BaseCaseMetadata, table=True):
    __tablename__ = "hk_case_metadata"
    # HK-specific: CACV / FACV / HCA citation code
    citation_code: str = Field(default="")


class HKLegisMetadata(BaseLegisMetadata, table=True):
    __tablename__ = "hk_legis_metadata"
    # HK Cap number extracted directly (e.g. "486")
    cap_number: str = Field(default="", index=True)
    cap_label:  str = Field(default="")              # "Cap 486"
    cap_type:   str = Field(default="")              # principal ordinance / subsidiary


# =============================================================================
# PAKISTAN  →  pak_case_metadata  |  pak_legis_metadata
# =============================================================================

class PakCaseMetadata(BaseCaseMetadata, table=True):
    __tablename__ = "pak_case_metadata"
    # Reporter: PLD | SCMR | PCrLJ | CLC
    case_reporter:       str = Field(default="", index=True)
    # Bench: full bench / division bench / single bench
    bench_composition:   str = Field(default="")
    # Law points decided (Pakistan judgment style)
    law_points_decided:  str = Field(default="", sa_type=Text)


class PakLegisMetadata(BaseLegisMetadata, table=True):
    __tablename__ = "pak_legis_metadata"
    # Federal vs provincial statute
    statute_level:  str = Field(default="")   # federal | provincial
    province:       str = Field(default="")   # Punjab | Sindh | KPK | Balochistan | Federal


# =============================================================================
# UNITED KINGDOM  →  uk_case_metadata  |  uk_legis_metadata
# =============================================================================

class UKCaseMetadata(BaseCaseMetadata, table=True):
    __tablename__ = "uk_case_metadata"
    # Neutral citation e.g. [2023] UKSC 14
    neutral_citation: str = Field(default="", index=True)
    # England/Wales | Scotland | Northern Ireland
    region:           str = Field(default="")


class UKLegisMetadata(BaseLegisMetadata, table=True):
    __tablename__ = "uk_legis_metadata"
    # Chapter number e.g. "c. 46" for Companies Act 2006
    chapter_number: str = Field(default="")
    # Statutory Instrument number e.g. "SI 2003/1234"
    si_number:      str = Field(default="")
    region:         str = Field(default="")  # UK-wide | England/Wales | Scotland | NI


# =============================================================================
# CANADA  →  ca_case_metadata  |  ca_legis_metadata
# =============================================================================

class CACaseMetadata(BaseCaseMetadata, table=True):
    __tablename__ = "ca_case_metadata"
    # Canada ingest uses a bigint identity, matching parent_id in Qdrant.
    id: int | None = Field(
        default=None,
        sa_column=Column(BigInteger, primary_key=True, autoincrement=True),
    )
    # Province / territory
    province:              str = Field(default="", index=True)
    # Proceeding language: en | fr | bilingual
    language_of_proceeding: str = Field(default="")


class CALegisMetadata(BaseLegisMetadata, table=True):
    __tablename__ = "ca_legis_metadata"
    # Revised Statutes reference e.g. "RSC 1985 c. C-46"
    rsc_reference: str = Field(default="")
    # Federal or provincial
    jurisdiction_level: str = Field(default="")  # federal | provincial
    province:           str = Field(default="")


# =============================================================================
# UNITED STATES  →  us_case_metadata  |  us_legis_metadata
# =============================================================================

class USCaseMetadata(BaseCaseMetadata, table=True):
    __tablename__ = "us_case_metadata"
    # Federal circuit: 1st–11th, DC, Federal | state
    circuit_or_state: str = Field(default="", index=True)
    # Reporter volume/page e.g. "547 U.S. 410"
    reporter_reference: str = Field(default="")


class USLegisMetadata(BaseLegisMetadata, table=True):
    __tablename__ = "us_legis_metadata"
    # US Code title e.g. "18" for Criminal Code
    usc_title:    str = Field(default="")
    # Public Law number e.g. "107-56"
    public_law_number: str = Field(default="")
    # CFR title for regulations
    cfr_title:    str = Field(default="")


# =============================================================================
# AUSTRALIA  →  au_case_metadata  |  au_legis_metadata
# =============================================================================

class AUCaseMetadata(BaseCaseMetadata, table=True):
    __tablename__ = "au_case_metadata"
    # Commonwealth or state
    jurisdiction_level: str = Field(default="")  # federal | state
    state_territory:    str = Field(default="", index=True)  # NSW | VIC | QLD | WA | SA | TAS | ACT | NT


class AULegisMetadata(BaseLegisMetadata, table=True):
    __tablename__ = "au_legis_metadata"
    # Cth | NSW | VIC | QLD | WA | SA | TAS | ACT | NT
    jurisdiction_level: str = Field(default="")
    state_territory:    str = Field(default="")


# =============================================================================
# INDIA  →  in_case_metadata  |  in_legis_metadata
# =============================================================================

class InCaseMetadata(BaseCaseMetadata, table=True):
    __tablename__ = "in_case_metadata"
    # High Court state / Supreme Court
    high_court_state: str = Field(default="", index=True)
    # AIR / SCC / SCR reporter
    case_reporter:    str = Field(default="", index=True)


class InLegisMetadata(BaseLegisMetadata, table=True):
    __tablename__ = "in_legis_metadata"
    # Central or State legislation
    jurisdiction_level: str = Field(default="")  # central | state
    state:              str = Field(default="")


# =============================================================================
# REGISTRIES  —  jurisdiction_key → SQLModel class
# =============================================================================

CASE_MODEL_REGISTRY: dict[str, Type[BaseCaseMetadata]] = {
    "hong_kong":     HKCaseMetadata,
    "pakistan":      PakCaseMetadata,
    "united_kingdom": UKCaseMetadata,
    "canada":        CACaseMetadata,
    "united_states": USCaseMetadata,
    "australia":     AUCaseMetadata,
    "india":         InCaseMetadata,
}

LEGIS_MODEL_REGISTRY: dict[str, Type[BaseLegisMetadata]] = {
    "hong_kong":     HKLegisMetadata,
    "pakistan":      PakLegisMetadata,
    "united_kingdom": UKLegisMetadata,
    "canada":        CALegisMetadata,
    "united_states": USLegisMetadata,
    "australia":     AULegisMetadata,
    "india":         InLegisMetadata,
}

# Table-name → registry-key (used for lookups from raw table names)
TABLE_TO_JURISDICTION: dict[str, str] = {
    model.__tablename__: jkey
    for jkey, model in {**CASE_MODEL_REGISTRY, **LEGIS_MODEL_REGISTRY}.items()
}


def get_case_model(jurisdiction_key: str) -> Type[BaseCaseMetadata]:
    model = CASE_MODEL_REGISTRY.get(jurisdiction_key)
    if not model:
        raise ValueError(f"No case model registered for jurisdiction_key='{jurisdiction_key}'")
    return model


def get_legis_model(jurisdiction_key: str) -> Type[BaseLegisMetadata]:
    model = LEGIS_MODEL_REGISTRY.get(jurisdiction_key)
    if not model:
        raise ValueError(f"No legislation model registered for jurisdiction_key='{jurisdiction_key}'")
    return model


def ensure_database_exists(database_url: str | None = None) -> None:
    """
    Connect to Postgres and create the database named in DATABASE_URL if missing.
    No-op for non-PostgreSQL URLs.
    """
    url = make_url(database_url or Config.DATABASE_URL)

    if not url.drivername.startswith("postgresql"):
        return

    db_name = url.database
    if not db_name:
        raise ValueError("DATABASE_URL must include a database name")

    admin_url = url.set(database="postgres")
    admin_engine = sa_create_engine(admin_url, isolation_level="AUTOCOMMIT")

    with admin_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": db_name},
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))


def init_db() -> None:
    """Create Postgres database (if needed) and all per-jurisdiction tables."""
    ensure_database_exists()
    SQLModel.metadata.create_all(engine)