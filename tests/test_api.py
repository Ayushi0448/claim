"""API contract, validation and error-handling tests."""

from __future__ import annotations

import pytest


class TestHealth:
    def test_health_reports_ready(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["index_ready"] is True
        assert body["chunks_indexed"] > 100
        assert body["policy_id"]
        assert "embedding" in body["backends"]

    def test_root_and_docs_available(self, api_client):
        assert api_client.get("/").status_code == 200
        assert api_client.get("/openapi.json").status_code == 200

    def test_index_info(self, api_client):
        body = api_client.get("/index/info").json()
        assert body["n_chunks"] > 100
        assert "What We Exclude" in body["sections"]
        assert body["sample"]


class TestAnalyzeContract:
    def test_returns_the_full_decision_contract(self, api_client, public_cases):
        response = api_client.post("/analyze", json=public_cases[0])
        assert response.status_code == 200
        body = response.json()
        for field in [
            "case_id", "decision", "confidence", "key_findings", "applicable_limits",
            "missing_evidence", "citations", "validation", "trace",
        ]:
            assert field in body, f"Response contract missing {field}"
        assert body["validation"]["status"] in {"PASS", "FAIL"}
        assert isinstance(body["validation"]["unsupported_claims"], list)

    def test_decision_is_one_of_the_five_statuses(self, api_client, public_cases):
        allowed = {
            "ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE",
            "NOT_ADMISSIBLE", "NEEDS_REVIEW",
        }
        for case in public_cases[:4]:
            body = api_client.post("/analyze", json=case).json()
            assert body["decision"] in allowed

    def test_confidence_in_range(self, api_client, public_cases):
        body = api_client.post("/analyze", json=public_cases[0]).json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_citations_carry_required_metadata(self, api_client, public_cases):
        body = api_client.post("/analyze", json=public_cases[0]).json()
        assert body["citations"]
        for citation in body["citations"]:
            assert citation["source"]
            assert isinstance(citation["page"], int) and citation["page"] >= 1
            assert citation["section"]
            assert citation["chunk_id"]
            assert citation["claim"]

    def test_wrapped_payload_accepted(self, api_client, public_cases):
        response = api_client.post("/analyze", json={"case": public_cases[0]})
        assert response.status_code == 200
        assert response.json()["case_id"] == public_cases[0]["case_id"]

    def test_evidence_and_trace_can_be_suppressed(self, api_client, public_cases):
        payload = {**public_cases[0], "include_evidence": False, "include_trace": False}
        body = api_client.post("/analyze", json=payload).json()
        assert body["evidence"] == []
        assert body["trace"] == []

    def test_batch_endpoint(self, api_client, public_cases):
        response = api_client.post(
            "/analyze/batch", json={"cases": public_cases[:3], "include_evidence": False}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 3
        assert len(body["results"]) == 3


class TestErrorHandling:
    def test_malformed_json(self, api_client):
        response = api_client.post(
            "/analyze", content=b"{not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in {400, 422}

    def test_missing_required_field(self, api_client):
        response = api_client.post("/analyze", json={"policy_id": "X"})
        assert response.status_code == 422
        assert "field_errors" in response.json()

    def test_blank_case_id_rejected(self, api_client):
        assert api_client.post("/analyze", json={"case_id": "   "}).status_code == 422

    @pytest.mark.parametrize(
        "payload",
        [
            {"case_id": "T", "sum_insured_inr": "not-a-number"},
            {"case_id": "T", "sum_insured_inr": -100},
            {"case_id": "T", "patient": {"age": 500}},
            {"case_id": "T", "treatment": {"admission_hours": -5}},
            {"case_id": "T", "documents": "should-be-a-list"},
        ],
    )
    def test_invalid_values_rejected(self, api_client, payload):
        assert api_client.post("/analyze", json=payload).status_code == 422

    def test_empty_body(self, api_client):
        assert api_client.post("/analyze", json={}).status_code == 422

    def test_null_body(self, api_client):
        assert api_client.post("/analyze", content=b"null",
                               headers={"Content-Type": "application/json"}).status_code == 422

    def test_unknown_fields_are_tolerated(self, api_client, public_cases):
        """RULE 9: irrelevant attributes must not cause rejection."""
        payload = {
            **public_cases[0],
            "loyalty_tier": "platinum",
            "referral_source": "app",
            "nested": {"arbitrary": ["values", 1, None]},
        }
        response = api_client.post("/analyze", json=payload)
        assert response.status_code == 200

    def test_irrelevant_attributes_do_not_change_the_decision(self, api_client, public_cases):
        base = api_client.post("/analyze", json=public_cases[0]).json()
        noisy = api_client.post(
            "/analyze",
            json={**public_cases[0], "vip": True, "agent_commission_inr": 9999,
                  "hospital_star_rating": 5},
        ).json()
        assert base["decision"] == noisy["decision"]
        assert base["total_deduction_inr"] == noisy["total_deduction_inr"]

    def test_minimal_case_abstains_rather_than_crashing(self, api_client):
        """A near-empty claim must produce a safe abstention, not a 500."""
        response = api_client.post("/analyze", json={"case_id": "MINIMAL-001"})
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "NEEDS_REVIEW"
        assert body["missing_evidence"]

    def test_oversized_batch_rejected(self, api_client, public_cases):
        response = api_client.post(
            "/analyze/batch", json={"cases": [public_cases[0]] * 51}
        )
        assert response.status_code == 422

    def test_wrong_method(self, api_client):
        assert api_client.get("/analyze").status_code == 405

    def test_unknown_route(self, api_client):
        assert api_client.get("/does-not-exist").status_code == 404


class TestDeterminism:
    def test_same_input_same_decision(self, api_client, public_cases):
        """Reproducibility: the pipeline is deterministic without an LLM."""
        first = api_client.post("/analyze", json=public_cases[6]).json()
        second = api_client.post("/analyze", json=public_cases[6]).json()
        assert first["decision"] == second["decision"]
        assert first["confidence"] == pytest.approx(second["confidence"])
        assert [c["chunk_id"] for c in first["citations"]] == [
            c["chunk_id"] for c in second["citations"]
        ]
