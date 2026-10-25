import hashlib
import itertools
import json
import threading

import pytest

from babble.harness.audit import GENESIS, AuditLog, TaggedAuditLog, verify_chain
from babble.types import Event


def _events() -> list[Event]:
    return [
        Event("llm_call", {"model": "m", "n": 1}, ts=1.0),
        Event("tool_call", {"name": "show_device", "input": {"device": "r1"}}, ts=2.0),
        Event("tool_result", {"is_error": False, "content": "ok"}, ts=3.0),
        Event("approval", {"approved": True, "tool": "configure_device"}, ts=4.0),
    ]


def _write_log(path, events=None) -> None:
    log = AuditLog(path)
    for event in events or _events():
        log.record(event)


def _lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _write_lines(path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------ in-memory log


def test_record_keeps_events_in_memory_without_path():
    log = AuditLog()
    for event in _events():
        log.record(event)
    assert log.events == _events()


def test_of_type_filters_by_type():
    log = AuditLog()
    for event in _events():
        log.record(event)
    approvals = log.of_type("approval")
    assert [e.data["tool"] for e in approvals] == ["configure_device"]
    assert log.of_type("stop") == []


# ------------------------------------------------------------- file format


def test_each_event_appends_one_json_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]
    assert [r["type"] for r in records] == ["llm_call", "tool_call", "tool_result", "approval"]
    assert records[0]["data"] == {"model": "m", "n": 1}
    assert records[0]["ts"] == 1.0


def test_first_line_links_to_genesis_and_each_line_links_to_previous(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    records = [json.loads(line) for line in _lines(path)]
    assert records[0]["prev_sha256"] == GENESIS == "0" * 64
    for prev, current in itertools.pairwise(records):
        assert current["prev_sha256"] == prev["sha256"]


def test_sha256_is_hash_of_canonical_record_without_sha256(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    for line in _lines(path):
        record = json.loads(line)
        claimed = record.pop("sha256")
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == claimed


def test_new_log_continues_chain_from_existing_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path, _events()[:2])
    second = AuditLog(path)
    for event in _events()[2:]:
        second.record(event)
    assert len(_lines(path)) == 4
    assert verify_chain(path) is True


# ----------------------------------------------------------- verify_chain


def test_intact_chain_verifies(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    assert verify_chain(path) is True


def test_empty_file_verifies(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text("", encoding="utf-8")
    assert verify_chain(path) is True


def test_single_line_chain_verifies(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path, _events()[:1])
    assert verify_chain(path) is True


def test_edited_field_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    record = json.loads(lines[1])
    record["data"]["name"] = "configure_device"
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    _write_lines(path, lines)
    assert verify_chain(path) is False


def test_edited_event_type_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    lines[0] = lines[0].replace('"llm_call"', '"stop"')
    _write_lines(path, lines)
    assert verify_chain(path) is False


def test_edit_with_recomputed_line_hash_is_still_detected(tmp_path):
    # An attacker who rehashes the edited line still breaks the next line's prev_sha256.
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    record = json.loads(lines[1])
    record["data"]["name"] = "configure_device"
    record.pop("sha256")
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    _write_lines(path, lines)
    assert verify_chain(path) is False


def test_edited_stored_sha256_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    record = json.loads(lines[2])
    record["sha256"] = "f" * 64
    lines[2] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    _write_lines(path, lines)
    assert verify_chain(path) is False


def test_deleted_middle_line_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    del lines[1]
    _write_lines(path, lines)
    assert verify_chain(path) is False


def test_deleted_first_line_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    del lines[0]
    _write_lines(path, lines)
    assert verify_chain(path) is False


def test_reordered_lines_are_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    lines[1], lines[2] = lines[2], lines[1]
    _write_lines(path, lines)
    assert verify_chain(path) is False


def test_swapping_first_and_last_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    lines[0], lines[-1] = lines[-1], lines[0]
    _write_lines(path, lines)
    assert verify_chain(path) is False


def test_inserted_blank_line_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    lines.insert(2, "")
    _write_lines(path, lines)
    assert verify_chain(path) is False


def test_garbage_line_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    _write_log(path)
    lines = _lines(path)
    lines[1] = "not json at all"
    _write_lines(path, lines)
    assert verify_chain(path) is False


UNHASHED_LINE = json.dumps({"type": "stop", "prev_sha256": GENESIS})


@pytest.mark.parametrize("bad", ['["a list"]', UNHASHED_LINE])
def test_non_object_or_unhashed_line_is_detected(tmp_path, bad):
    path = tmp_path / "audit.jsonl"
    path.write_text(bad + "\n", encoding="utf-8")
    assert verify_chain(path) is False


def test_concurrent_records_keep_the_chain_intact(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    per_thread, thread_count = 50, 8
    barrier = threading.Barrier(thread_count)

    def record_many(worker: int) -> None:
        barrier.wait()
        for n in range(per_thread):
            log.record(Event("tool_result", {"worker": worker, "n": n}))

    threads = [threading.Thread(target=record_many, args=(w,)) for w in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(log.events) == per_thread * thread_count
    assert len(path.read_text(encoding="utf-8").splitlines()) == per_thread * thread_count
    assert verify_chain(path)


def test_tagged_log_forwards_events_with_tags_merged_into_data():
    parent = AuditLog()
    tagged = TaggedAuditLog(parent, subagent="r3")
    original = Event("tool_call", {"name": "show", "input": {"command": "show version"}}, ts=7.5)

    tagged.record(original)

    (forwarded,) = parent.events
    assert forwarded.type == "tool_call"
    assert forwarded.ts == 7.5
    assert forwarded.data == {
        "name": "show",
        "input": {"command": "show version"},
        "subagent": "r3",
    }
    # The caller's event is not mutated.
    assert original.data == {"name": "show", "input": {"command": "show version"}}


def test_tagged_logs_from_two_children_share_one_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    parent = AuditLog(path)
    TaggedAuditLog(parent, subagent="r1").record(Event("llm_call", {"turn": 1}))
    TaggedAuditLog(parent, subagent="r2").record(Event("llm_call", {"turn": 1}))

    assert [e.data["subagent"] for e in parent.events] == ["r1", "r2"]
    assert verify_chain(path)
