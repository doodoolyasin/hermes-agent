import pytest
from gateway.platforms.event import MessageEvent


def test_bidi_and_rtl_marks_slash_command_extraction():
    # Persian RTL keyboards often insert LRM (\u200e) or RLM (\u200f) before /
    cases = [
        ("\u200e/model openai-fast", "model", "openai-fast"),
        ("\u200f/status", "status", ""),
        ("\ufeff/diagnose", "diagnose", ""),
        ("\u200b/help", "help", ""),
        ("   /clear   ", "clear", ""),
        ("\u200e\u200f/model gpt-4o", "model", "gpt-4o"),
        ("/model@HermesBot openai-fast", "model", "openai-fast"),
        ("\u200e/model@MyBaleBot gpt-4o", "model", "gpt-4o"),
    ]

    for text, expected_cmd, expected_args in cases:
        evt = MessageEvent(text=text)
        assert evt.is_command() is True, f"Failed is_command for {text!r}"
        assert evt.get_command() == expected_cmd, f"Wrong command for {text!r}"
        assert evt.get_command_args() == expected_args, f"Wrong args for {text!r}"


def test_slash_command_fuzzing_non_commands():
    non_commands = [
        "hello /world",
        "https://example.com/api",
        "/etc/passwd",
        "/",
        "///",
        "",
        None,
        "   ",
        "\u200eسلام چطوری؟",
    ]

    for text in non_commands:
        evt = MessageEvent(text=text)
        assert evt.get_command() is None
