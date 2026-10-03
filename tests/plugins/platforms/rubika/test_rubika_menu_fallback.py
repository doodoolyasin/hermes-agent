import pytest
from gateway.config import PlatformConfig
from plugins.platforms.rubika.adapter import RubikaAdapter


def test_rubika_menu_rendering_and_persian_digits_selection():
    cfg = PlatformConfig(extra={"token": "fake"}, enabled=True)
    adapter = RubikaAdapter(cfg)

    options = ["وضعیت سرور", "تغییر مدل", "کمک و راهنما", "تنظیمات"]
    rendered = adapter.render_fallback_menu("منوی اصلی هرمس", options)

    assert "منوی اصلی هرمس" in rendered
    assert "1️⃣ وضعیت سرور" in rendered
    assert "2️⃣ تغییر مدل" in rendered
    assert "3️⃣ کمک و راهنما" in rendered
    assert "4️⃣ تنظیمات" in rendered

    # 1. Standard ASCII numbers (1-based -> 0-based)
    assert adapter.parse_menu_choice("1", options) == 0
    assert adapter.parse_menu_choice("2", options) == 1
    assert adapter.parse_menu_choice("4", options) == 3

    # 2. Persian numerals (۱, ۲, ۳, ۴)
    assert adapter.parse_menu_choice("۱", options) == 0
    assert adapter.parse_menu_choice("۲", options) == 1
    assert adapter.parse_menu_choice("۳", options) == 2
    assert adapter.parse_menu_choice("۴", options) == 3

    # 3. Arabic numerals (١, ٢, ٣, ٤)
    assert adapter.parse_menu_choice("١", options) == 0
    assert adapter.parse_menu_choice("٣", options) == 2

    # 4. Text label matching
    assert adapter.parse_menu_choice("وضعیت سرور", options) == 0
    assert adapter.parse_menu_choice("  تغییر مدل  ", options) == 1
    assert adapter.parse_menu_choice("تنظیمات", options) == 3

    # 5. Out of bounds & invalid inputs
    assert adapter.parse_menu_choice("0", options) is None
    assert adapter.parse_menu_choice("۰", options) is None
    assert adapter.parse_menu_choice("5", options) is None
    assert adapter.parse_menu_choice("۵", options) is None
    assert adapter.parse_menu_choice("-1", options) is None
    assert adapter.parse_menu_choice("ناموجود", options) is None
    assert adapter.parse_menu_choice("", options) is None
    assert adapter.parse_menu_choice(None, options) is None
