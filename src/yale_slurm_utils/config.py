"""User configuration: where settings/themes live, and how they persist.

Also handles best-effort detection of the terminal's background colour so we
can pick a sensible light/dark default theme.
"""

from __future__ import annotations

import json
import os
import re
import select
import sys
import time
from pathlib import Path


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    path = Path(base) / "yale-slurm-utils"
    path.mkdir(parents=True, exist_ok=True)
    return path


def themes_dir() -> Path:
    path = config_dir() / "themes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _settings_path() -> Path:
    return config_dir() / "config.json"


def load_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_settings(settings: dict) -> None:
    _settings_path().write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def get_theme_preference() -> str | None:
    """The theme name the user chose, or ``None`` (meaning auto-detect)."""
    value = load_settings().get("theme")
    return value or None


def set_theme_preference(name: str | None) -> None:
    settings = load_settings()
    if name is None or name == "auto":
        settings.pop("theme", None)
    else:
        settings["theme"] = name
    save_settings(settings)


# --------------------------------------------------------------------------- #
# Terminal background detection
# --------------------------------------------------------------------------- #
def _query_osc11(timeout: float = 0.15) -> bool | None:
    """Ask the terminal for its background colour via OSC 11.

    Returns True if the background looks dark, False if light, None if we
    couldn't tell (no TTY / unsupported terminal).
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    try:
        import termios
        import tty
    except ImportError:  # pragma: no cover - non-POSIX
        return None

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except (termios.error, ValueError):  # pragma: no cover - env dependent
        return None

    buf = ""
    try:
        tty.setraw(fd)
        sys.stdout.write("\x1b]11;?\x07")
        sys.stdout.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            ready, _, _ = select.select([fd], [], [], max(0.0, remaining))
            if not ready:
                break
            buf += os.read(fd, 64).decode(errors="ignore")
            if "\x07" in buf or "\x1b\\" in buf:
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    match = re.search(r"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", buf)
    if not match:
        return None

    def _component(hex_str: str) -> float:
        value = int(hex_str, 16)
        scale = (1 << (len(hex_str) * 4)) - 1
        return value / scale if scale else 0.0

    r, g, b = (_component(match.group(i)) for i in (1, 2, 3))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance < 0.5


def detect_dark_background() -> bool | None:
    """True if the terminal background is dark, False if light, None if unknown."""
    # COLORFGBG (set by many terminals) is cheap and doesn't touch the TTY.
    cfb = os.environ.get("COLORFGBG")
    if cfb and ";" in cfb:
        try:
            bg = int(cfb.split(";")[-1])
            # 7 (white) and 15 (bright white) are light backgrounds.
            return bg not in (7, 15)
        except ValueError:
            pass
    return _query_osc11()
