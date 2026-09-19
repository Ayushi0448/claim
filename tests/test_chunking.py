"""Chunking and ingestion tests."""

from __future__ import annotations

import re

from app.retrieval.chunking import embedding_text
from app.utils.text import contains_phrase


class TestIngestion:
    def test_all_pages_ingested(self, policy_pages):
        assert len(policy_pages) == 17, "The supplied policy has 17 pages"
        assert all(p.page == i + 1 for i, p in enumerate(policy_pages))

    def test_boilerplate_is_stripped(self, policy_pages):
        """The running header must not survive into the indexed text."""
        header_hits = sum(
            1 for p in policy_pages
            if contains_phrase(p.text, "UNIVERSAL SOMPO GENERAL INSURANCE CO LTD")
        )
        assert header_hits == 0, "Running header leaked into chunk text"

    def test_substantive_text_survives(self, policy_pages):
        combined = " ".join(p.text for p in policy_pages)
        assert contains_phrase(combined, "Pre-existing diseases will not be covered until 48 months")
        assert contains_phrase(combined, "Normal Room expenses: 1.0% of Basic Sum Insured")


class TestChunkMetadata:
    def test_every_chunk_is_fully_attributed(self, chunks):
        """Provenance is what makes a citation checkable; none may be missing."""
        for chunk in chunks:
            assert chunk.chunk_id, "chunk_id missing"
            assert chunk.text.strip(), f"{chunk.chunk_id} has empty text"
            assert 1 <= chunk.page <= 17, f"{chunk.chunk_id} has page {chunk.page}"
            assert chunk.section, f"{chunk.chunk_id} has no section"
            assert chunk.heading, f"{chunk.chunk_id} has no heading"
            assert chunk.source, f"{chunk.chunk_id} has no source"
            assert chunk.char_count == len(chunk.text)
            assert chunk.token_estimate > 0

    def test_chunk_ids_are_unique(self, chunks):
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_encodes_page(self, chunks):
        for chunk in chunks:
            match = re.match(r"^p(\d{2})-", chunk.chunk_id)
            assert match, f"{chunk.chunk_id} does not encode its page"
            assert int(match.group(1)) == chunk.page

    def test_size_bounds(self, chunks, settings):
        for chunk in chunks:
            assert chunk.char_count <= settings.chunking.max_chunk_chars + 200
            assert chunk.char_count >= 40


class TestStructureAwareness:
    def test_expected_sections_detected(self, chunks):
        sections = {c.section for c in chunks}
        for expected in {
            "Definitions", "Scope of Cover", "What We Exclude",
            "Standard Terms and Conditions", "Claims Procedure",
        }:
            assert expected in sections, f"Section not detected: {expected}"

    def test_not_naive_fixed_size(self, chunks):
        """A fixed-size splitter produces near-uniform lengths; this must not."""
        lengths = [c.char_count for c in chunks]
        spread = max(lengths) - min(lengths)
        assert spread > 500, "Chunk sizes look uniform, suggesting fixed-size splitting"
        assert len({c.heading for c in chunks}) > 50, "Headings are not being derived per unit"

    def test_definitions_are_atomic(self, chunks):
        """Each defined term should be its own chunk, headed by the term."""
        definitions = [c for c in chunks if c.section == "Definitions"]
        assert len(definitions) >= 40
        headings = {c.heading.lower() for c in definitions}
        for term in ["hospital", "domiciliary treatment", "medically necessary"]:
            assert any(term in h for h in headings), f"Definition not isolated: {term}"

    def test_exclusions_are_numbered_units(self, chunks):
        exclusions = [c for c in chunks if c.section == "What We Exclude" and c.clause_ref]
        refs = {c.clause_ref for c in exclusions}
        # The supplied policy has 21 numbered exclusions.
        assert len(refs) >= 18, f"Only {len(refs)} numbered exclusions isolated"
        assert "Exclusion 1" in refs and "Exclusion 7" in refs

    def test_decisive_clauses_are_intact_and_on_the_right_page(self, chunks):
        """Each governing clause must survive chunking whole, on its real page."""
        required = [
            ("Pre-existing diseases will not be covered until 48 months", 8),
            ("waiting period of 30 days will apply to all claims", 9),
            ("Normal Room expenses: 1.0% of Basic Sum Insured", 7),
            ("Intensive Care/ Therapeutic Unit expenses: 2% of Basic Sum Insured", 7),
            ("subject to a limit of 25% of Sum Assured", 7),
            ("subject to a limit of 40% Sum Insured", 7),
            ("maximum aggregate sub-limit of 20% of the Basic Sum Insured", 7),
            ("Pre-Hospitalisation up to a maximum of 30 days", 8),
            ("Hospital means any institution established for in-patient care", 3),
            ("Domiciliary Treatment means medical treatment", 2),
            ("Unproven/Experimental Treatment means", 6),
            ("cosmetic or aesthetic treatment of any description", 9),
            ("Dental treatment or surgery of any kind", 9),
            ("140 Day Care Procedures", 7),
        ]
        for phrase, page in required:
            hits = [c for c in chunks if contains_phrase(c.text, phrase)]
            assert hits, f"Clause lost during chunking: {phrase!r}"
            assert any(c.page == page for c in hits), (
                f"{phrase!r} found on pages {[c.page for c in hits]}, expected {page}"
            )


class TestEmbeddingText:
    def test_context_prefix_added_for_indexing_only(self, chunks):
        chunk = chunks[10]
        text = embedding_text(chunk)
        assert chunk.section in text and chunk.heading in text
        assert chunk.text in text
        # The stored text stays pristine so citations quote the policy verbatim.
        assert not chunk.text.startswith(chunk.section)
