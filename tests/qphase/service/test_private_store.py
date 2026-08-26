"""UserPrivateStore: private tags, saved views and project locations."""

from __future__ import annotations

import pytest
from qphase.service.private import UserPrivateStore


def test_private_tags_roundtrip(tmp_path):
    store = UserPrivateStore("project-a", home=tmp_path)

    store.add_private_tag("session", "s1", "task:wip")
    store.add_private_tag("session", "s2", "task:wip")
    store.add_private_tag("session", "s1", "purpose:draft")

    assert store.list_private_tags("session") == [
        ("s1", "task:wip"),
        ("s2", "task:wip"),
        ("s1", "purpose:draft"),
    ]
    assert store.list_private_tags("session", "s1") == [
        ("s1", "task:wip"),
        ("s1", "purpose:draft"),
    ]

    store.remove_private_tag("session", "s1", "task:wip")
    assert store.list_private_tags("session", "s1") == [("s1", "purpose:draft")]


def test_two_project_ids_do_not_pollute_each_other(tmp_path):
    first = UserPrivateStore("project-a", home=tmp_path)
    second = UserPrivateStore("project-b", home=tmp_path)

    first.add_private_tag("session", "s1", "task:wip")
    first.save_view("mine", {"object_kind": "session"})

    assert second.list_private_tags("session") == []
    assert second.list_views() == []
    assert first.path != second.path
    assert first.path.parent == second.path.parent


def test_reads_on_missing_database_do_not_create_files(tmp_path):
    store = UserPrivateStore("project-a", home=tmp_path)

    assert store.list_private_tags("session") == []
    assert store.list_views() == []
    assert store.list_locations() == []
    store.remove_private_tag("session", "s1", "task:wip")
    store.delete_view("mine")
    assert not store.path.exists()


def test_saved_views_roundtrip(tmp_path):
    store = UserPrivateStore("project-a", home=tmp_path)

    store.save_view("review", {"object_kind": "session", "lifecycle": "active"})
    store.save_view("review", {"object_kind": "artifact"})
    store.save_view("cold", {"object_kind": "session", "lifecycle": "archived"})

    assert store.list_views() == [
        ("cold", {"object_kind": "session", "lifecycle": "archived"}),
        ("review", {"object_kind": "artifact"}),
    ]

    store.delete_view("review")
    assert [name for name, _ in store.list_views()] == ["cold"]


def test_project_locations_roundtrip(tmp_path):
    store = UserPrivateStore("project-a", home=tmp_path)

    store.record_location(tmp_path / "somewhere")
    store.record_location(tmp_path / "elsewhere")

    locations = store.list_locations()
    assert [root for _pid, root, _seen in locations] == [
        str(tmp_path / "elsewhere")
    ]
    assert all(pid == "project-a" for pid, _root, _seen in locations)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
