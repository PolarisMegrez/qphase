"""qphase_cam: the search-evidence assembler
---------------------------------------------------------
One root service turns :class:`~qphase_cam.evidence.SearchAuditFragment` records
into the mandatory search-evidence product. It is **not** a plugin: the engine
calls it directly, because every search task must publish evidence whether or
not a user selected anything, and a selectable "evidence plugin" would make that
contract optional.

What the assembler does — and does not — do:

* It consumes fragments, an optional candidate product and catalog references,
  and calls :func:`~qphase_cam.data.evidence.build_evidence_product`. It never
  computes a scientific quantity: counts, coverage, identities and near misses
  arrive already stated by the solver.
* It validates what the builder cannot know: that every input is a fragment,
  that case identities are unique, that one search task does not silently mix
  two search identities, and that a source artifact is a catalog id rather than
  a file path.
* A summary never re-derives a fragment. An evidence product is read back
  through :func:`fragments_from_evidence_product` (the symmetric reader), and a
  candidate-only artifact is read through
  :func:`fragments_from_candidate_product`, which reuses the fragment the solver
  stored for each case. Guessing an event from counts is not an option.

``EvidenceAssembler(allow_multiple_searches=True)`` is what a ``summarize_search``
job uses: several upstream searches may be summarized together, and the product
then records one design per search identity instead of pretending they share
one. A ``search`` job uses the default, so two different searches cannot end up
in one evidence product by accident.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qphase.core.utils import canonical_json
from qphase.data import Dataset

from .data.evidence import (
    SourceArtifactRef,
    build_evidence_product,
    fragments_from_candidate_product,
    fragments_from_evidence_product,
)
from .evidence import SearchAuditFragment

__all__ = [
    "CANDIDATE_PRODUCT",
    "EVIDENCE_PRODUCT",
    "EvidenceAssembler",
    "artifact_reference",
]

#: Product role of the candidate product a search produces.
CANDIDATE_PRODUCT = "candidates"

#: Product role of the evidence product this service assembles.
EVIDENCE_PRODUCT = "evidence"


def artifact_reference(result: Any) -> SourceArtifactRef:
    """Return the catalog reference of one upstream result.

    The reference is the artifact id the artifact manifest published — carried
    by a restored bundle in its metadata and by an in-memory bundle in its
    provenance — never a directory or a job path. A result without one cannot
    be cited, so it fails instead of being referenced by location.
    """
    artifact_id = _artifact_id(result)
    if artifact_id is None:
        raise ValueError(
            "an upstream result carries no artifact_id, so the summary cannot "
            "cite it; a summary references published artifacts, not paths"
        )
    return SourceArtifactRef(artifact_id=artifact_id)


def _artifact_id(result: Any) -> str | None:
    """Return the published artifact id of one result, if it states one."""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping):
        declared = metadata.get("artifact_id")
        if isinstance(declared, str) and declared.strip():
            return declared
    provenance = getattr(result, "provenance", None)
    if isinstance(provenance, Mapping):
        declared = provenance.get("artifact_id")
        if isinstance(declared, str) and declared.strip():
            return declared
    return None


class EvidenceAssembler:
    """Build one evidence product from solver-owned audit fragments."""

    def __init__(self, *, allow_multiple_searches: bool = False) -> None:
        self.allow_multiple_searches = bool(allow_multiple_searches)

    # -- entry points -----------------------------------------------------

    def from_fragments(
        self,
        fragments: Sequence[SearchAuditFragment],
        *,
        artifact_id: str,
        candidate_product: Dataset | None = None,
        source_artifacts: Sequence[SourceArtifactRef | Mapping[str, Any]] = (),
        search_design: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Dataset:
        """Assemble the evidence product of one search (or one summary).

        ``artifact_id`` is the artifact this product is published into; it is
        added as a self reference so the evidence can cite the artifact that
        carries it. Extra ``source_artifacts`` are upstream artifacts (a summary
        cites the searches it summarizes).
        """
        cases = self._validated_fragments(fragments)
        design = self._search_design(cases, search_design)
        references = self._references(artifact_id, source_artifacts)
        return build_evidence_product(
            fragments=cases,
            source_artifacts=references,
            search_design=design,
            candidate_product=candidate_product,
            provenance={
                "artifact_id": str(artifact_id),
                "evidence_roles": {
                    "candidates": CANDIDATE_PRODUCT,
                    "evidence": EVIDENCE_PRODUCT,
                },
                **dict(provenance or {}),
            },
        )

    def from_products(
        self,
        products: Mapping[str, Dataset],
        *,
        artifact_id: str,
        candidate_product: Dataset | None = None,
        source_artifacts: Sequence[SourceArtifactRef | Mapping[str, Any]] = (),
        search_design: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Dataset:
        """Assemble evidence from already-produced CAM products.

        An evidence product wins over the candidate product it was derived
        from: reading both would count the same case twice. A candidate-only
        mapping is read through the fragments the solver stored per case.
        """
        fragments = self.fragments_from_products(products)
        candidate = candidate_product
        if candidate is None:
            candidates = products.get(CANDIDATE_PRODUCT)
            if isinstance(candidates, Dataset):
                candidate = candidates
        return self.from_fragments(
            fragments,
            artifact_id=artifact_id,
            candidate_product=candidate,
            source_artifacts=source_artifacts,
            search_design=search_design,
            provenance=provenance,
        )

    def fragments_from_products(
        self, products: Mapping[str, Dataset]
    ) -> tuple[SearchAuditFragment, ...]:
        """Return the fragments one already-produced product mapping states."""
        evidence = products.get(EVIDENCE_PRODUCT)
        if isinstance(evidence, Dataset):
            return fragments_from_evidence_product(evidence)
        candidates = products.get(CANDIDATE_PRODUCT)
        if isinstance(candidates, Dataset):
            return fragments_from_candidate_product(candidates)
        raise ValueError(
            "no search evidence can be assembled from these products: the "
            f"mapping has {sorted(products)}, but it needs an {EVIDENCE_PRODUCT!r} "
            f"or a {CANDIDATE_PRODUCT!r} product"
        )

    # -- validation -------------------------------------------------------

    def _validated_fragments(
        self, fragments: Sequence[SearchAuditFragment]
    ) -> list[SearchAuditFragment]:
        """Return the cases after checking type, uniqueness and search scope."""
        cases: list[SearchAuditFragment] = []
        for fragment in fragments:
            if not isinstance(fragment, SearchAuditFragment):
                raise TypeError(
                    "the evidence assembler consumes SearchAuditFragment objects; "
                    f"got {type(fragment).__name__}"
                )
            cases.append(fragment)
        if not cases:
            raise ValueError(
                "a search must publish evidence: no audit fragment was produced"
            )
        identities = [case.case_identity for case in cases]
        duplicated = sorted({name for name in identities if identities.count(name) > 1})
        if duplicated:
            raise ValueError(
                f"case identities must be unique in one evidence product; "
                f"duplicated: {duplicated}"
            )
        searches = sorted({case.search_identity for case in cases})
        if len(searches) > 1 and not self.allow_multiple_searches:
            raise ValueError(
                f"one search task must not mix search identities {searches}; a "
                "summary that combines several searches states it explicitly"
            )
        return cases

    def _search_design(
        self,
        cases: Sequence[SearchAuditFragment],
        declared: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Return the design attribute of the product.

        A single search states one design. A summary over several searches keeps
        one design per search identity, so the attribute never claims that
        different targets or orders were one design.
        """
        if declared is not None:
            return dict(declared)
        per_case: dict[str, Any] = {}
        for case in cases:
            per_case.setdefault(canonical_json(dict(case.search_design)), case)
        if len(per_case) == 1:
            return dict(cases[0].search_design)
        by_search: dict[str, Any] = {}
        for case in cases:
            existing = by_search.setdefault(
                case.search_identity, dict(case.search_design)
            )
            if canonical_json(existing) != canonical_json(dict(case.search_design)):
                raise ValueError(
                    f"search {case.search_identity!r} states two search designs; a "
                    "case cannot belong to one search and two designs"
                )
        return {"per_search_design": by_search}

    @staticmethod
    def _references(
        artifact_id: str,
        extra: Sequence[SourceArtifactRef | Mapping[str, Any]],
    ) -> list[SourceArtifactRef]:
        """Return the source artifacts: the self reference plus upstream ones."""
        references: list[SourceArtifactRef] = [
            SourceArtifactRef(artifact_id=artifact_id)
        ]
        for value in extra:
            reference = (
                value
                if isinstance(value, SourceArtifactRef)
                else SourceArtifactRef.model_validate(value)
            )
            if any(
                existing.artifact_id == reference.artifact_id for existing in references
            ):
                continue
            references.append(reference)
        return references
