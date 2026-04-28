from src.preprocessing import clean_text


def test_clean_text_removes_noise():
    text = "Help needed @rescue now! https://example.com 😢"
    out = clean_text(text)
    assert "http" not in out
    assert "@rescue" not in out
    assert "help" in out
