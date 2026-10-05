"""
jurisdiction_registry.py — Multi-Jurisdiction Registry for Kanooni Jawab
==================================================================
Central source of truth for all supported legal jurisdictions.
Provides:
  • JurisdictionConfig — per-jurisdiction citation patterns, field mappings,
    vocabulary hints, and metadata schema keys
  • JURISDICTION_REGISTRY — all supported jurisdictions
  • normalize_jurisdiction()  — canonical name resolution
  • get_jurisdiction_config() — safe config lookup
  • build_routing_regexes()   — dynamic regex compilation per jurisdiction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class JurisdictionConfig:
    """Complete configuration for one legal jurisdiction."""

    # Canonical display name stored in Qdrant payloads
    canonical_name: str

    # All aliases that map → canonical_name (lowercase)
    aliases: list[str] = field(default_factory=list)

    # Court / citation reference codes used in regex routing
    case_citation_codes: list[str] = field(default_factory=list)

    # Legislation-specific keywords common to this jurisdiction
    legislation_keywords: list[str] = field(default_factory=list)

    # Qdrant payload key that stores the legislation identifier
    # (e.g. "cap_number" for HK, "act_number" for Pakistan/UK/Canada)
    legislation_id_key: str = "act_number"

    # Regex pattern to extract the legislation number from a query string
    # Group 1 must capture the identifier value
    legislation_id_pattern: str = r"\bact\s+no\.?\s*(\d+[\w]*)\b"

    # Labels that appear in Qdrant payloads alongside the legislation id
    # e.g. ["Cap 486", "Cap. 486"] for HK, ["Act No. 1"] for Pakistan
    legislation_label_prefixes: list[str] = field(default_factory=list)

    # Additional Qdrant payload keys to try when filtering legislation
    extra_legislation_keys: list[str] = field(default_factory=list)

    # Vocabulary hints injected into the Gemini #1 prompt so sub-queries
    # use jurisdiction-appropriate terminology
    legal_vocab_hints: list[str] = field(default_factory=list)

    # Language(s) used in legal documents (for chunking / OCR hints)
    languages: list[str] = field(default_factory=list)

    # Court hierarchy labels (informational; used in structured queries)
    court_hierarchy: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------

JURISDICTION_REGISTRY: dict[str, JurisdictionConfig] = {

    # ── Hong Kong ────────────────────────────────────────────────────────────
    "hong_kong": JurisdictionConfig(
        canonical_name="Hong Kong",
        aliases=[
            "hong kong", "hk", "hksar", "hong kong sar",
            "hong kong special administrative region",
        ],
        case_citation_codes=[
            "FCJA", "CACV", "FACC", "HCA", "DCCJ", "CMP",
            "CACJ", "FACV", "HCAL", "HCMP", "DCCC", "ESCC",
        ],
        legislation_keywords=[
            "cap", "cap.", "ordinance", "schedule", "subsidiary legislation",
            "gazette", "chapter",
        ],
        legislation_id_key="cap_number",
        legislation_id_pattern=r"\bcap\.?\s*(\d+[A-Za-z]?)\b",
        legislation_label_prefixes=["Cap", "Cap."],
        extra_legislation_keys=["cap_label", "legislation_number"],
        legal_vocab_hints=[
            "Cap. number", "ordinance", "subsidiary legislation",
            "Court of Final Appeal", "Court of Appeal", "High Court",
            "District Court", "Magistrates Court",
        ],
        languages=["en", "zh"],
        court_hierarchy=[
            "Court of Final Appeal", "Court of Appeal",
            "Court of First Instance", "District Court",
            "Magistrates Court", "Lands Tribunal",
        ],
    ),

    # ── Pakistan ─────────────────────────────────────────────────────────────
    "pakistan": JurisdictionConfig(
        canonical_name="Pakistan",
        aliases=[
            "pakistan", "pak", "islamic republic of pakistan",
            "pakistan sar",
        ],
        case_citation_codes=[
            "SCMR", "PLD", "PCrLJ", "PTD", "CLC",
            "MLD", "CLCN", "YLR", "PLJ", "NLR",
            "PSC", "PLC", "PL",
        ],
        legislation_keywords=[
            "act", "ordinance", "regulation", "rule", "section",
            "schedule", "statute", "code", "crpc", "ppc", "qanun",
            "hudood", "zina", "hadd", "tazir",
        ],
        legislation_id_key="act_number",
        legislation_id_pattern=(
            r"\bact\s+(?:no\.?\s*)?(\d+[\w]*)\s+of\s+(\d{4})\b"
            r"|\bord(?:inance)?\s+(?:no\.?\s*)?(\w+)\s+of\s+(\d{4})\b"
        ),
        legislation_label_prefixes=["Act", "Ordinance", "Rule"],
        extra_legislation_keys=["statute_id", "ordinance_number"],
        legal_vocab_hints=[
            "PLD", "SCMR", "Supreme Court", "Lahore High Court",
            "Sindh High Court", "Peshawar High Court", "Islamabad High Court",
            "Balochistan High Court", "Federal Shariat Court",
            "Sessions Court", "qanun-e-shahadat", "CrPC", "PPC",
        ],
        languages=["en", "ur"],
        court_hierarchy=[
            "Supreme Court of Pakistan",
            "Federal Shariat Court",
            "Lahore High Court", "Sindh High Court",
            "Peshawar High Court", "Islamabad High Court",
            "Balochistan High Court",
            "Sessions Court", "Civil Court", "Judicial Magistrate",
        ],
    ),

    # ── United Kingdom ────────────────────────────────────────────────────────
    "united_kingdom": JurisdictionConfig(
        canonical_name="United Kingdom",
        aliases=[
            "united kingdom", "uk", "great britain", "britain",
            "england and wales", "england", "wales", "scotland",
            "northern ireland", "england & wales",
        ],
        case_citation_codes=[
            "UKSC", "UKHL", "EWCA", "EWHC", "EWCOP",
            "EWFC", "AC", "QB", "KB", "Ch", "Fam",
            "CSIH", "CSOH", "NICA", "NIQB", "WLR",
        ],
        legislation_keywords=[
            "act", "statute", "statutory instrument", "si", "section",
            "schedule", "regulations", "order", "directive",
            "chapter", "c.", "sch",
        ],
        legislation_id_key="act_number",
        legislation_id_pattern=(
            r"\bact\s+(\d{4})\b"
            r"|\bc\.\s*(\d+)\b"
            r"|\bsi\s+(\d{4}/\d+)\b"
        ),
        legislation_label_prefixes=["Act", "SI", "S.I.", "c."],
        extra_legislation_keys=["statute_id", "chapter_number"],
        legal_vocab_hints=[
            "Supreme Court", "Court of Appeal", "High Court",
            "Crown Court", "Magistrates Court", "Tribunal",
            "statutory instrument", "secondary legislation",
            "Hansard", "Royal Assent",
        ],
        languages=["en"],
        court_hierarchy=[
            "Supreme Court of the United Kingdom",
            "Court of Appeal", "High Court of Justice",
            "Crown Court", "County Court", "Magistrates Court",
            "First-tier Tribunal", "Upper Tribunal",
        ],
    ),

    # ── Canada ────────────────────────────────────────────────────────────────
    "canada": JurisdictionConfig(
        canonical_name="Canada",
        aliases=[
            "canada", "can", "canadian", "dominion of canada",
            "ontario", "british columbia", "alberta", "quebec",
            "nova scotia", "new brunswick", "manitoba",
            "saskatchewan", "newfoundland", "prince edward island",
        ],
        case_citation_codes=[
            "SCC", "FCA", "FC", "ONCA", "ONSC", "BCCA",
            "BCSC", "ABCA", "ABKB", "QCCA", "QCCS",
            "SCR", "DLR", "OR", "BCLR", "AR", "NBR",
            "NSCA", "NSSC", "MBCA", "MBQB", "SKCA", "SKQB",
        ],
        legislation_keywords=[
            "act", "code", "regulation", "statute", "rsc",
            "rso", "sbc", "sa", "ccsm", "section", "schedule",
            "consolidated", "revised statutes",
        ],
        legislation_id_key="act_number",
        legislation_id_pattern=(
            r"\bRSC\s+(\d{4})\b"
            r"|\bSBC\s+(\d{4})\b"
            r"|\bRSO\s+(\d{4})\b"
            r"|\bact\s+(\d{4})\b"
        ),
        legislation_label_prefixes=["RSC", "SBC", "RSO", "SA", "Act"],
        extra_legislation_keys=["statute_id", "rsc_reference"],
        legal_vocab_hints=[
            "Supreme Court of Canada", "Federal Court of Appeal",
            "RSC", "provincial statute", "Charter of Rights",
            "Criminal Code", "Civil Code of Quebec",
            "bilingual legislation",
        ],
        languages=["en", "fr"],
        court_hierarchy=[
            "Supreme Court of Canada",
            "Federal Court of Appeal", "Federal Court",
            "Provincial Court of Appeal", "Superior Court",
            "Provincial Court",
        ],
    ),

    # ── United States ─────────────────────────────────────────────────────────
    "united_states": JurisdictionConfig(
        canonical_name="United States",
        aliases=[
            "united states", "usa", "us", "america",
            "united states of america",
        ],
        case_citation_codes=[
            "US", "S.Ct", "L.Ed", "F.3d", "F.4th", "F.Supp",
            "F.Supp.2d", "F.Supp.3d", "WL", "LEXIS",
        ],
        legislation_keywords=[
            "usc", "u.s.c", "cfr", "code", "section", "§",
            "statute", "act", "public law", "pl", "title",
        ],
        legislation_id_key="act_number",
        legislation_id_pattern=(
            r"\b(\d+)\s+U\.?S\.?C\.?\s*§\s*(\d+)\b"
            r"|\bPub\.\s*L\.\s*(\d+-\d+)\b"
        ),
        legislation_label_prefixes=["USC", "CFR", "Pub. L."],
        extra_legislation_keys=["statute_id", "title_number"],
        legal_vocab_hints=[
            "Supreme Court", "Circuit Court", "District Court",
            "USC", "CFR", "due process", "equal protection",
            "federal circuit", "en banc",
        ],
        languages=["en"],
        court_hierarchy=[
            "Supreme Court of the United States",
            "United States Court of Appeals",
            "United States District Court",
            "United States Bankruptcy Court",
            "United States Tax Court",
        ],
    ),

    # ── Australia ─────────────────────────────────────────────────────────────
    "australia": JurisdictionConfig(
        canonical_name="Australia",
        aliases=[
            "australia", "aus", "commonwealth of australia",
            "new south wales", "nsw", "victoria", "vic",
            "queensland", "qld", "western australia", "wa",
            "south australia", "sa", "tasmania", "tas",
            "act", "northern territory", "nt",
        ],
        case_citation_codes=[
            "HCA", "FCAFC", "FCA", "NSWCA", "NSWSC",
            "VSC", "VSCA", "QCA", "QSC", "WASC", "WASCA",
            "ALR", "CLR", "ALJR",
        ],
        legislation_keywords=[
            "act", "regulation", "legislative instrument",
            "commonwealth", "state act", "section", "schedule",
        ],
        legislation_id_key="act_number",
        legislation_id_pattern=r"\bact\s+(\d{4})\b|\bno\.\s*(\d+)\s+of\s+(\d{4})\b",
        legislation_label_prefixes=["Act", "No."],
        extra_legislation_keys=["statute_id"],
        legal_vocab_hints=[
            "High Court of Australia", "Federal Court",
            "Commonwealth statute", "Cth", "legislative instrument",
        ],
        languages=["en"],
        court_hierarchy=[
            "High Court of Australia",
            "Full Federal Court", "Federal Court of Australia",
            "State Supreme Court", "District Court",
            "Magistrates Court",
        ],
    ),

    # ── India ─────────────────────────────────────────────────────────────────
    "india": JurisdictionConfig(
        canonical_name="India",
        aliases=[
            "india", "ind", "republic of india", "bharat",
        ],
        case_citation_codes=[
            "SCC", "AIR", "SCR", "SC", "HC", "Bom", "Cal",
            "Mad", "Del", "All", "Ker", "Guj", "MP", "Raj",
        ],
        legislation_keywords=[
            "act", "code", "section", "schedule", "ipc", "crpc",
            "constitution", "article", "amendment", "ordinance",
        ],
        legislation_id_key="act_number",
        legislation_id_pattern=r"\bact\s+(?:no\.?\s*)?(\d+)\s+of\s+(\d{4})\b",
        legislation_label_prefixes=["Act", "Ord."],
        extra_legislation_keys=["statute_id"],
        legal_vocab_hints=[
            "Supreme Court of India", "High Court", "IPC", "CrPC",
            "AIR", "SCC", "District Court", "Sessions Court",
        ],
        languages=["en", "hi"],
        court_hierarchy=[
            "Supreme Court of India",
            "High Court", "Sessions Court",
            "Civil Court", "Magistrate Court",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Alias → canonical key lookup (built once at import time)
# ---------------------------------------------------------------------------

_ALIAS_TO_KEY: dict[str, str] = {}
for _jkey, _jcfg in JURISDICTION_REGISTRY.items():
    for _alias in _jcfg.aliases:
        _ALIAS_TO_KEY[_alias.lower().strip()] = _jkey
    # The canonical_name itself also resolves
    _ALIAS_TO_KEY[_jcfg.canonical_name.lower().strip()] = _jkey
    # The registry key itself
    _ALIAS_TO_KEY[_jkey.lower().strip()] = _jkey


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_jurisdiction(value: str) -> str:
    """
    Convert any alias or variant name to the canonical display name.

    Examples:
        "hk"        → "Hong Kong"
        "pak"       → "Pakistan"
        "england"   → "United Kingdom"
        "ontario"   → "Canada"
        "unknown"   → "unknown"   (returned as-is, lowercased)
    """
    key = (value or "").strip().lower()
    registry_key = _ALIAS_TO_KEY.get(key)
    if registry_key:
        return JURISDICTION_REGISTRY[registry_key].canonical_name
    return value.strip() if value else ""


def get_jurisdiction_key(canonical_or_alias: str) -> Optional[str]:
    """Return the registry key (e.g. 'hong_kong') or None if not found."""
    key = (canonical_or_alias or "").strip().lower()
    return _ALIAS_TO_KEY.get(key)


def get_jurisdiction_config(canonical_or_alias: str) -> Optional[JurisdictionConfig]:
    """Return the JurisdictionConfig for a name/alias, or None if unknown."""
    jkey = get_jurisdiction_key(canonical_or_alias)
    return JURISDICTION_REGISTRY.get(jkey) if jkey else None


def list_canonical_names() -> list[str]:
    """All supported canonical jurisdiction names."""
    return [cfg.canonical_name for cfg in JURISDICTION_REGISTRY.values()]


# ---------------------------------------------------------------------------
# Dynamic regex builders
# ---------------------------------------------------------------------------

def build_case_routing_regex(jurisdiction_key: str) -> Optional[re.Pattern]:
    """
    Build a compiled case-citation regex for a specific jurisdiction.
    Returns None if the jurisdiction is unknown or has no codes.
    """
    cfg = JURISDICTION_REGISTRY.get(jurisdiction_key)
    if not cfg or not cfg.case_citation_codes:
        return None
    patterns = [rf"\b{re.escape(code)}\b" for code in cfg.case_citation_codes]
    patterns += [
        r"\bapplicant\b", r"\brespondent\b", r"\bpetitioner\b",
        r"\bdefendant\b", r"\bplaintiff\b", r"\bappellant\b",
        r"\bclaimant\b", r"\bjudgment\b", r"\bv\.\s*[A-Z0-9]",
        r"\bcase\s+no\.?\b", r"\bcourt\s+of\s+appeal\b",
    ]
    return re.compile("|".join(patterns), re.IGNORECASE)


def build_legislation_routing_regex(jurisdiction_key: str) -> Optional[re.Pattern]:
    """
    Build a compiled legislation-routing regex for a specific jurisdiction.
    """
    cfg = JURISDICTION_REGISTRY.get(jurisdiction_key)
    if not cfg:
        return None
    kws = cfg.legislation_keywords or []
    patterns = [rf"\b{re.escape(kw)}\b" for kw in kws]
    # Add the legislation_id_pattern itself as a routing signal
    if cfg.legislation_id_pattern:
        patterns.append(cfg.legislation_id_pattern)
    # Generic fallbacks
    patterns += [
        r"\bsection\s+\d+", r"\bschedule\b",
        r"\bstatute\b", r"\blegislation\b",
    ]
    return re.compile("|".join(patterns), re.IGNORECASE)


def extract_legislation_id(query: str, jurisdiction_key: str) -> Optional[str]:
    """
    Extract the primary legislation identifier from a query string
    using the jurisdiction-specific pattern.

    Returns the first non-empty captured group, or None.
    """
    cfg = JURISDICTION_REGISTRY.get(jurisdiction_key)
    if not cfg or not cfg.legislation_id_pattern:
        return None
    match = re.search(cfg.legislation_id_pattern, query or "", re.IGNORECASE)
    if not match:
        return None
    # Return first non-None group
    for g in match.groups():
        if g:
            return g.upper().strip()
    return None


def get_legislation_qdrant_conditions(
    query: str,
    jurisdiction_key: str,
) -> list[dict]:
    """
    Build a list of Qdrant FieldCondition-compatible dicts for legislation
    filtering, based on the jurisdiction's payload schema.

    The caller should convert these into actual qmodels.FieldCondition objects.
    Each dict: {"key": str, "value": str}
    """
    cfg = JURISDICTION_REGISTRY.get(jurisdiction_key)
    if not cfg:
        return []

    leg_id = extract_legislation_id(query, jurisdiction_key)
    if not leg_id:
        return []

    conditions: list[dict] = [
        {"key": cfg.legislation_id_key, "value": leg_id}
    ]

    for prefix in cfg.legislation_label_prefixes:
        conditions.append({"key": cfg.legislation_id_key, "value": f"{prefix} {leg_id}"})
        conditions.append({"key": cfg.legislation_id_key, "value": f"{prefix}. {leg_id}"})

    for extra_key in cfg.extra_legislation_keys:
        conditions.append({"key": extra_key, "value": leg_id})
        for prefix in cfg.legislation_label_prefixes:
            conditions.append({"key": extra_key, "value": f"{prefix} {leg_id}"})

    return conditions


def get_vocab_hint_block(jurisdiction_key: str) -> str:
    """
    Return a formatted string of jurisdiction-specific legal vocabulary,
    suitable for injecting into the Gemini #1 prompt.
    """
    cfg = JURISDICTION_REGISTRY.get(jurisdiction_key)
    if not cfg or not cfg.legal_vocab_hints:
        return ""
    hints = ", ".join(cfg.legal_vocab_hints)
    return (
        f"Jurisdiction-specific legal vocabulary for {cfg.canonical_name}:\n"
        f"  {hints}\n"
        f"Use these terms naturally in sub-query expansions when relevant."
    )