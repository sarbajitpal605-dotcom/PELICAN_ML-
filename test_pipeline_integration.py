"""End-to-end integration tests for personal memory classification and retrieval pipeline.

Chains together the four security & classification stages:
  1. secret_filter.py   (Deterministic secret pre-filter)
  2. classify_memory.py (LLM category and direct-statement classification)
  3. content_version.py (Hash freshness and usability resolution)
  4. authorization.py   (Request-time client authorization and one-time grant verification)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any
import pytest

from authorization import authorize_recall
from classify_memory import classify_memory, compute_content_hash
from content_version import resolve_memory_for_use
from secret_filter import scan_for_secrets


class FakeLLMClient:
    """Mock LLM client returning canned JSON responses without network calls."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        if not self.responses:
            raise RuntimeError("FakeLLMClient: No remaining canned responses")
        return self.responses.pop(0)


class ExplosiveLLMClient:
    """Mock LLM client designed to explode if called, proving early pipeline halts."""

    def complete(self, prompt: str) -> str:
        raise AssertionError("ExplosiveLLMClient.complete() was invoked when pipeline should have halted early!")


# ============================================================================
# Simulated Ingestion Pipeline Helper
# ============================================================================

def process_memory_ingestion(
    memory_id: str,
    raw_text: str,
    llm_client: Any,
    db: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Simulate end-to-end ingestion pipeline storing into an in-memory database."""
    # STAGE A: Pre-filter for sensitive secrets
    secret_result = scan_for_secrets(raw_text)
    if secret_result["blocked"]:
        record = {
            "memory_id": memory_id,
            "current_text": raw_text,
            "status": "blocked",
            "classification": None,
            "blocked_patterns": secret_result["matched_patterns"],
        }
        db[memory_id] = record
        return record

    # STAGE B: LLM Classification
    classification_res = classify_memory(raw_text, llm_client)
    if classification_res is None:
        record = {
            "memory_id": memory_id,
            "current_text": raw_text,
            "status": "failed",
            "classification": None,
        }
    else:
        record = {
            "memory_id": memory_id,
            "current_text": raw_text,
            "status": "ready",
            "classification": classification_res.model_dump(),
        }

    db[memory_id] = record
    return record


# ============================================================================
# Scenario 1 - Normal general fact, full success path
# ============================================================================

def test_scenario_1_normal_general_fact_success_path() -> None:
    """Scenario 1: Normal general fact walks through all 4 stages to allowed recall."""
    memory_db: dict[str, dict[str, Any]] = {}
    memory_id = "mem_sc1_001"
    raw_text = "I use TypeScript for this project."
    client_id = "ext_client_1"
    fixed_now = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)

    # STAGE A: Secret filter
    secret_res = scan_for_secrets(raw_text)
    assert not secret_res["blocked"], f"Expected clean text to pass, got: {secret_res}"
    assert secret_res["matched_patterns"] == []

    # STAGE B: Classification
    canned_llm_json = json.dumps({
        "category": "general",
        "confidence": 0.98,
        "is_direct_statement": True,
        "reasoning": "Standard technical programming stack preference.",
    })
    llm_client = FakeLLMClient([canned_llm_json])
    record = process_memory_ingestion(memory_id, raw_text, llm_client, memory_db)

    assert record["status"] == "ready"
    assert record["classification"] is not None
    assert record["classification"]["category"] == "general"
    assert record["classification"]["is_direct_statement"] is True
    assert record["classification"]["content_hash"] == compute_content_hash(raw_text)
    assert llm_client.call_count == 1

    # STAGE C: Content version resolution
    resolution = resolve_memory_for_use(record)
    assert resolution["available"] is True, f"Expected memory to be available, got: {resolution}"
    assert resolution["reason"] == "ok"
    assert resolution.get("requires_approval") is False

    # STAGE D: Authorization recall (grant=None because general is auto-injectable)
    auth_result = authorize_recall(record, client_id, grant=None, now=fixed_now)
    assert auth_result["allowed"] is True, f"Expected recall to be allowed, got: {auth_result}"
    assert auth_result["reason"] == "ok"


# ============================================================================
# Scenario 2 - Sensitive fact requiring approval, full grant lifecycle
# ============================================================================

def test_scenario_2_sensitive_fact_grant_lifecycle() -> None:
    """Scenario 2: Sensitive fact requiring approval, verifying full one-time grant lifecycle."""
    memory_db: dict[str, dict[str, Any]] = {}
    memory_id = "mem_sc2_001"
    raw_text = "I was prescribed medication for anxiety last month."
    client_id = "ext_client_1"
    fixed_now = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)

    # STAGE A: Secret filter
    secret_res = scan_for_secrets(raw_text)
    assert not secret_res["blocked"], "Medical note should pass deterministic secret regex"

    # STAGE B: Classification
    canned_llm_json = json.dumps({
        "category": "sensitive_uncertain",
        "confidence": 0.99,
        "is_direct_statement": True,
        "reasoning": "Personal medical diagnosis and prescription details.",
    })
    llm_client = FakeLLMClient([canned_llm_json])
    record = process_memory_ingestion(memory_id, raw_text, llm_client, memory_db)

    assert record["status"] == "ready"
    assert record["classification"] is not None
    assert record["classification"]["category"] == "sensitive_uncertain"
    assert record["classification"]["is_direct_statement"] is True

    # STAGE C: Content version resolution
    resolution = resolve_memory_for_use(record)
    assert resolution["available"] is True
    assert resolution["reason"] == "requires_approval"
    assert resolution.get("requires_approval") is True

    # STAGE D, sub-case (i): grant=None -> denied
    auth_sub_i = authorize_recall(record, client_id, grant=None, now=fixed_now)
    assert auth_sub_i["allowed"] is False
    assert auth_sub_i["reason"] == "grant_required", f"Expected grant_required, got: {auth_sub_i}"

    # STAGE D, sub-case (ii): valid unconsumed unexpired grant -> allowed with consume_grant=True
    grant = {
        "memory_id": memory_id,
        "client_id": client_id,
        "granted_at": fixed_now - timedelta(minutes=2),
        "expires_at": fixed_now + timedelta(minutes=10),
        "consumed": False,
    }
    auth_sub_ii = authorize_recall(record, client_id, grant=grant, now=fixed_now)
    assert auth_sub_ii["allowed"] is True
    assert auth_sub_ii["reason"] == "ok"
    assert auth_sub_ii.get("consume_grant") is True

    # STAGE D, sub-case (iii): re-run with SAME grant marked consumed=True -> denied (replay prevented)
    grant["consumed"] = True  # Caller marks it consumed after sub-case ii
    auth_sub_iii = authorize_recall(record, client_id, grant=grant, now=fixed_now)
    assert auth_sub_iii["allowed"] is False
    assert auth_sub_iii["reason"] == "grant_already_consumed", (
        f"Replay attack failed to block; expected grant_already_consumed, got: {auth_sub_iii}"
    )


# ============================================================================
# Scenario 3 - Obvious secret caught by the regex pre-filter
# ============================================================================

def test_scenario_3_obvious_secret_blocked_at_prefilter() -> None:
    """Scenario 3: Obvious AWS key is blocked at Stage A; downstream stages are NEVER invoked."""
    memory_db: dict[str, dict[str, Any]] = {}
    memory_id = "mem_sc3_001"
    raw_text = "My AWS key is AKIAIOSFODNN7EXAMPLE"

    # STAGE A: Secret filter
    secret_res = scan_for_secrets(raw_text)
    assert secret_res["blocked"] is True
    assert "aws_access_key" in secret_res["matched_patterns"]

    # Provide an explosive LLM client that will raise an error if invoked
    explosive_llm = ExplosiveLLMClient()

    # Ingestion halts immediately at Stage A without calling LLM or downstream stages
    record = process_memory_ingestion(memory_id, raw_text, explosive_llm, memory_db)

    assert record["status"] == "blocked"
    assert record["classification"] is None
    assert memory_db[memory_id]["status"] == "blocked"

    # Stage C resolution check confirms blocked status is unavailable
    resolution = resolve_memory_for_use(record)
    assert resolution["available"] is False
    assert resolution["reason"] == "status_blocked"

    # Stage D authorization recall confirms blocked memory cannot be recalled
    fixed_now = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
    auth_result = authorize_recall(record, "ext_client_1", grant=None, now=fixed_now)
    assert auth_result["allowed"] is False
    assert auth_result["reason"] == "status_blocked"


# ============================================================================
# Scenario 4 - Edited memory invalidates a previously-general classification
# ============================================================================

def test_scenario_4_edited_memory_invalidates_classification() -> None:
    """Scenario 4: Edited memory text invalidates classification end-to-end via hash mismatch."""
    memory_db: dict[str, dict[str, Any]] = {}
    memory_id = "mem_sc4_001"
    original_text = "I use TypeScript for this project."
    client_id = "ext_client_1"
    fixed_now = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)

    # Ingest Scenario 1 text as ready & general
    canned_llm_json = json.dumps({
        "category": "general",
        "confidence": 0.98,
        "is_direct_statement": True,
        "reasoning": "Developer tooling preference.",
    })
    record = process_memory_ingestion(memory_id, original_text, FakeLLMClient([canned_llm_json]), memory_db)
    assert record["status"] == "ready"
    assert record["classification"]["category"] == "general"

    # Initial recall works cleanly
    initial_auth = authorize_recall(record, client_id, grant=None, now=fixed_now)
    assert initial_auth["allowed"] is True
    assert initial_auth["reason"] == "ok"

    # Mutate current_text WITHOUT reclassifying (simulate user edit in UI)
    record["current_text"] = "I use Rust for this project and have moved away from TypeScript."

    # Stage C: Content version resolution rejects stale classification
    resolution_after_edit = resolve_memory_for_use(record)
    assert resolution_after_edit["available"] is False
    assert resolution_after_edit["reason"] == "stale_classification"

    # Stage D: Final authorization gate denies recall with the exact same stale reason
    auth_after_edit = authorize_recall(record, client_id, grant=None, now=fixed_now)
    assert auth_after_edit["allowed"] is False
    assert auth_after_edit["reason"] == "stale_classification", (
        f"Expected stale_classification to deny authorization, got: {auth_after_edit}"
    )

    # Prove stored classification object still said 'general' (preventing silent reuse)
    assert record["classification"]["category"] == "general"
