"""Tests for the theme system: partial YAML, persistence, auto-detect, version."""

from __future__ import annotations

import importlib.metadata

import pytest

from yale_slurm_utils import __version__, config, theme


@pytest.fixture()
def user_config(tmp_path, monkeypatch):
    """Point config at a throwaway directory."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Make detection deterministic unless a test overrides it.
    monkeypatch.setattr(config, "detect_dark_background", lambda: True)
    return tmp_path


def test_version_single_source():
    assert __version__ == importlib.metadata.version("yale-slurm-utils")


def test_builtins_have_light_and_dark():
    modes = {t.mode for t in theme.BUILTIN_THEMES.values()}
    assert {"light", "dark"} <= modes


def test_theme_from_dict_is_partial():
    t = theme.theme_from_dict({"accent": "#123456"}, name="mini")
    assert t.name == "mini"
    assert t.accent == "#123456"
    # Untouched keys keep the dark defaults.
    assert t.idle == theme.Theme().idle
    # Unknown keys are ignored, not fatal.
    t2 = theme.theme_from_dict({"bogus": "x", "mode": "light"})
    assert t2.mode == "light"


def test_user_yaml_theme_roundtrip(user_config):
    themes_dir = config.themes_dir()
    (themes_dir / "midnight.yaml").write_text(
        "name: midnight\nmode: dark\naccent: '#5aa2ff'\n", encoding="utf-8"
    )
    loaded = theme.load_user_themes()
    assert "midnight" in loaded
    assert loaded["midnight"].accent == "#5aa2ff"
    assert "midnight" in theme.all_themes()


def test_preference_roundtrip(user_config):
    assert config.get_theme_preference() is None
    config.set_theme_preference("nord")
    assert config.get_theme_preference() == "nord"
    assert theme.resolve_theme().name == "nord"
    # "auto" clears the preference.
    config.set_theme_preference("auto")
    assert config.get_theme_preference() is None


def test_resolve_auto_uses_background(user_config, monkeypatch):
    monkeypatch.setattr(config, "detect_dark_background", lambda: False)
    assert theme.resolve_theme("auto").mode == "light"
    monkeypatch.setattr(config, "detect_dark_background", lambda: True)
    assert theme.resolve_theme("auto").mode == "dark"


def test_resolve_unknown_name_falls_back(user_config):
    # Doesn't raise; picks a sane default instead.
    assert theme.resolve_theme("does-not-exist") is not None


def test_apply_switches_active_styles():
    theme.set_active(theme.BUILTIN_THEMES["dark"])
    dark_border = theme.border()
    theme.set_active(theme.BUILTIN_THEMES["light"])
    assert theme.border() != dark_border
    # restore default for other tests
    theme.set_active(theme.BUILTIN_THEMES["dark"])
