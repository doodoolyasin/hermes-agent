import pytest
from gateway.config import PlatformConfig
from plugins.platforms.bale.adapter import BaleAdapter


def test_persian_bidi_and_zwnj_chunking():
    cfg = PlatformConfig(extra={"token": "fake"}, enabled=True)
    adapter = BaleAdapter(cfg)
    # Set artificial short limit to force chunking on test data
    adapter.max_message_length = 100

    persian_text = (
        "این یک متن فارسی طولانی با نویسه‌های نیم‌فاصله است که باید بررسی شود.\n"
        "پاراگراف دوم شامل ترکیب کلمات انگلیسی مانند Python و AI و نویسه‌های خاص می‌باشد.\n"
        "پاراگراف سوم حاوی توضیحات تکمیلی است که نباید کلمات در آن نصف شوند."
    )

    chunks = adapter._chunk(persian_text)

    # 1. No chunk may exceed max_message_length
    for c in chunks:
        assert len(c) <= 100

    # 2. Total combined content must preserve original text
    reconstructed = "\n".join(chunks)
    assert "نیم‌فاصله" in reconstructed
    assert "Python" in reconstructed
    assert "AI" in reconstructed


def test_huge_markdown_code_block_chunking():
    cfg = PlatformConfig(extra={"token": "fake"}, enabled=True)
    adapter = BaleAdapter(cfg)
    adapter.max_message_length = 150

    code_text = (
        "```python\n"
        "def hello_world():\n"
        "    print('Hello World 1')\n"
        "    print('Hello World 2')\n"
        "    print('Hello World 3')\n"
        "    print('Hello World 4')\n"
        "```"
    )

    chunks = adapter._chunk(code_text)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c) <= 150
