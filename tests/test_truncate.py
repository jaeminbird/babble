import pytest

from babbler.harness.truncate import truncate


def test_short_text_is_unchanged():
    assert truncate("abc", 3) == "abc"
    assert truncate("", 0) == ""


def test_long_text_keeps_head_and_tail_with_marker():
    text = "H" * 100 + "M" * 100 + "T" * 100  # 300 chars
    out = truncate(text, 100)
    head, rest = out.split("\n[...", 1)
    assert head == "H" * 60
    assert rest.endswith("]\n" + "T" * 40)
    dropped = 300 - 60 - 40
    assert f"[... {dropped} characters truncated." in out


def test_marker_names_how_to_narrow():
    out = truncate("x" * 500, 50)
    assert "Narrow the command" in out
    assert "neighbor, prefix or interface" in out


def test_result_keeps_the_start_and_end_of_the_output():
    text = "".join(str(i % 10) for i in range(1000))
    out = truncate(text, 200)
    assert out.startswith(text[:120])
    assert out.endswith(text[-80:])


def test_zero_limit_drops_everything_but_the_marker():
    out = truncate("abcdef", 0)
    assert out.startswith("\n[... 6 characters truncated.")


def test_negative_limit_is_rejected():
    with pytest.raises(ValueError):
        truncate("abc", -1)
