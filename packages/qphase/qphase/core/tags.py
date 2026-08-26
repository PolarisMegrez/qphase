"""Tag syntax, canonicalization and the project tag policy.

Tags are the user-facing organization layer of qphase objects. A tag has the
form ``namespace:value`` or ``namespace:path/to/value``; the namespace is
governed by the project tag policy (``configs/tags.yaml``, schema
``qphase.tag-policy/1``). Validation happens in two phases:

1. workflow load — pure syntax canonicalization
   (:func:`canonicalize_tag_syntax`), no policy involved;
2. workflow compilation / submission / annotation writes — full policy
   validation (:meth:`TagPolicy.apply`): namespace existence, cardinality,
   allowed values, object applicability, aliases and reserved namespaces.

Without a policy file the project has no tag governance: tags are
syntax-canonicalized only and the policy revision is ``None``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import ErrorCode, QPhaseConfigError
from .project import ProjectContext
from .utils import canonical_json, load_yaml

__all__ = [
    "TAG_POLICY_FILENAME",
    "TAG_POLICY_SCHEMA",
    "ObjectKind",
    "TagNamespacePolicy",
    "TagPolicy",
    "canonicalize_tag_syntax",
    "load_tag_policy",
    "parse_tag",
]

#: Schema identifier of the project tag policy document.
TAG_POLICY_SCHEMA: Literal["qphase.tag-policy/1"] = "qphase.tag-policy/1"

#: Policy file location relative to the project defaults directory.
TAG_POLICY_FILENAME = "tags.yaml"

ObjectKind = Literal[
    "workflow", "job", "execution", "session", "artifact", "occurrence"
]
OBJECT_KINDS: tuple[str, ...] = (
    "workflow",
    "job",
    "execution",
    "session",
    "artifact",
    "occurrence",
)

_NAMESPACE_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_SEGMENT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")


def parse_tag(value: str) -> tuple[str, str]:
    """Split a tag into ``(namespace, value_path)`` after syntax checks."""
    text = value.strip()
    namespace, separator, path = text.partition(":")
    if not separator:
        raise QPhaseConfigError(
            f"tag {value!r} must have the form 'namespace:value'",
            code=ErrorCode.CONFIG,
        )
    namespace = namespace.lower()
    if not _NAMESPACE_PATTERN.fullmatch(namespace):
        raise QPhaseConfigError(
            f"tag namespace {namespace!r} must match [a-z][a-z0-9_]*",
            code=ErrorCode.CONFIG,
        )
    segments = tuple(
        segment.lower() for segment in path.split("/") if segment.strip()
    )
    if not segments or "/".join(segments) != path.strip().lower():
        raise QPhaseConfigError(
            f"tag {value!r} contains empty or padded path segments",
            code=ErrorCode.CONFIG,
        )
    for segment in segments:
        if not _SEGMENT_PATTERN.fullmatch(segment):
            raise QPhaseConfigError(
                f"tag path segment {segment!r} must match [a-z0-9][a-z0-9_.-]*",
                code=ErrorCode.CONFIG,
            )
    return namespace, "/".join(segments)


def canonicalize_tag_syntax(value: str) -> str:
    """Return the canonical spelling of one tag (syntax phase only)."""
    namespace, path = parse_tag(value)
    return f"{namespace}:{path}"


def canonicalize_tag_list(values: list[str]) -> list[str]:
    """Syntax-canonicalize a declared tag list; duplicates are rejected."""
    canonical = [canonicalize_tag_syntax(value) for value in values]
    duplicates = sorted({tag for tag in canonical if canonical.count(tag) > 1})
    if duplicates:
        raise QPhaseConfigError(
            f"duplicate tags after canonicalization: {duplicates}",
            code=ErrorCode.CONFIG,
        )
    return canonical


class TagNamespacePolicy(BaseModel):
    """Governance rules of one tag namespace."""

    model_config = ConfigDict(extra="forbid")

    hierarchical: bool = True
    cardinality: Literal["one", "many"] = "many"
    open: bool = False
    values: tuple[str, ...] = ()
    aliases: dict[str, str] = Field(default_factory=dict)
    objects: tuple[ObjectKind, ...] = ()
    inherit: bool = True


class TagPolicy(BaseModel):
    """The project tag policy (``configs/tags.yaml``).

    Unknown namespaces are rejected; free-form tags are only possible inside
    namespaces explicitly declared ``open``. ``revision`` is a content hash so
    snapshots record exactly which policy validated them.
    """

    model_config = ConfigDict(extra="forbid")

    schema_: Literal["qphase.tag-policy/1"] = Field(
        default=TAG_POLICY_SCHEMA, alias="schema"
    )
    namespaces: dict[str, TagNamespacePolicy] = Field(default_factory=dict)
    reserved_namespaces: tuple[str, ...] = ()
    retention_inherits_to_occurrences: bool = True

    @property
    def revision(self) -> str:
        """Content-hash revision of the policy for snapshot freezing."""
        payload = self.model_dump(mode="json", by_alias=True)
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]

    def apply(self, values: list[str], object_kind: ObjectKind) -> list[str]:
        """Validate and canonicalize tags for one object against the policy."""
        canonical = []
        for value in values:
            tag = canonicalize_tag_syntax(value)
            namespace, path = parse_tag(tag)
            rule = self.namespaces.get(namespace)
            if namespace in self.reserved_namespaces:
                raise QPhaseConfigError(
                    f"tag namespace {namespace!r} is reserved",
                    code=ErrorCode.CONFIG,
                )
            if rule is None:
                raise QPhaseConfigError(
                    f"unknown tag namespace {namespace!r}; declare it in "
                    "configs/tags.yaml or use an open namespace",
                    code=ErrorCode.CONFIG,
                )
            alias = rule.aliases.get(path)
            if alias is None and "/" in path:
                head, _, rest = path.partition("/")
                head_alias = rule.aliases.get(head)
                if head_alias is not None:
                    alias = f"{head_alias}/{rest}"
            if alias is not None:
                _, path = parse_tag(f"{namespace}:{alias}")
                tag = f"{namespace}:{path}"
            if rule.objects and object_kind not in rule.objects:
                raise QPhaseConfigError(
                    f"tag namespace {namespace!r} does not apply to "
                    f"{object_kind} objects",
                    code=ErrorCode.CONFIG,
                )
            if not rule.hierarchical and "/" in path:
                raise QPhaseConfigError(
                    f"tag namespace {namespace!r} is not hierarchical: {tag!r}",
                    code=ErrorCode.CONFIG,
                )
            if not rule.open and not self._value_allowed(rule, path):
                raise QPhaseConfigError(
                    f"tag {tag!r} is not an allowed value of namespace "
                    f"{namespace!r}",
                    code=ErrorCode.CONFIG,
                )
            canonical.append(tag)
        seen: dict[str, str] = {}
        for tag in canonical:
            namespace = tag.split(":", 1)[0]
            rule = self.namespaces[namespace]
            if rule.cardinality == "one" and namespace in seen:
                raise QPhaseConfigError(
                    f"namespace {namespace!r} allows one tag per object, got "
                    f"{seen[namespace]!r} and {tag!r}",
                    code=ErrorCode.CONFIG,
                )
            seen[namespace] = tag
        if len(set(canonical)) != len(canonical):
            raise QPhaseConfigError(
                "duplicate tags after canonicalization",
                code=ErrorCode.CONFIG,
            )
        return canonical

    def allows_inheritance(self, tag: str) -> bool:
        """Return whether the tag's namespace participates in inheritance."""
        namespace = tag.split(":", 1)[0]
        rule = self.namespaces.get(namespace)
        return rule.inherit if rule is not None else True

    def tag_applies_to(self, tag: str, object_kind: ObjectKind) -> bool:
        """Return whether the tag's namespace covers the object kind."""
        namespace = tag.split(":", 1)[0]
        rule = self.namespaces.get(namespace)
        if rule is None or not rule.objects:
            return True
        return object_kind in rule.objects

    @staticmethod
    def _value_allowed(rule: TagNamespacePolicy, path: str) -> bool:
        for allowed in rule.values:
            _, candidate = parse_tag(f"x:{allowed}")
            if path == candidate:
                return True
            if rule.hierarchical and path.startswith(candidate + "/"):
                return True
        return False


def load_tag_policy(project: ProjectContext) -> TagPolicy | None:
    """Load the project tag policy; ``None`` when the project has none."""
    path = project.defaults_path.parent / TAG_POLICY_FILENAME
    if not path.exists():
        return None
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise QPhaseConfigError(
            f"tag policy {path} must be a mapping",
            code=ErrorCode.CONFIG,
        )
    try:
        return TagPolicy.model_validate(data)
    except Exception as exc:
        raise QPhaseConfigError(
            f"invalid tag policy {path}: {exc}",
            code=ErrorCode.CONFIG,
        ) from exc


def validate_declared_tags(
    values: list[str], object_kind: ObjectKind, policy: TagPolicy | None
) -> list[str]:
    """Single entry point: policy validation when present, else syntax only."""
    if policy is None:
        return canonicalize_tag_list(values)
    return policy.apply(values, object_kind)
