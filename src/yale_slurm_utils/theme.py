"""Shared colours, styles and small visual helpers (Yale blue flavoured)."""

from __future__ import annotations

from rich.text import Text

# Yale blue: #00356B. Used for headers / branding accents.
YALE_BLUE = "#00356b"
YALE_BLUE_BRIGHT = "#286dc0"

# A stable palette so each GPU model keeps the same colour across views.
_GPU_PALETTE = [
    "bright_green",
    "bright_magenta",
    "bright_cyan",
    "bright_yellow",
    "bright_blue",
    "bright_red",
    "green",
    "magenta",
]
_GPU_COLOR_CACHE: dict[str, str] = {}


def gpu_color(gpu_type: str) -> str:
    if gpu_type not in _GPU_COLOR_CACHE:
        idx = len(_GPU_COLOR_CACHE) % len(_GPU_PALETTE)
        _GPU_COLOR_CACHE[gpu_type] = _GPU_PALETTE[idx]
    return _GPU_COLOR_CACHE[gpu_type]


def state_style(state: str) -> str:
    s = state.lower()
    if "idle" in s:
        return "green"
    if "mix" in s:
        return "yellow"
    if "alloc" in s or "running" in s:
        return "bright_red"
    if "pending" in s:
        return "bright_yellow"
    if any(t in s for t in ("down", "fail", "drain", "maint")):
        return "red dim"
    if "complet" in s:
        return "cyan"
    return "white"


def util_style(pct: float) -> str:
    """Colour for a utilisation/occupancy percentage (busy == hot)."""
    if pct >= 90:
        return "bright_red"
    if pct >= 66:
        return "yellow"
    if pct >= 33:
        return "green"
    return "bright_green"


def availability_style(free: int, total: int) -> str:
    """Colour for *free* counts (more free == cooler/greener)."""
    if total == 0:
        return "dim"
    ratio = free / total
    if free == 0:
        return "bright_red"
    if ratio <= 0.15:
        return "yellow"
    return "bright_green"


def usage_bar(
    used: int,
    total: int,
    width: int = 20,
    *,
    invert: bool = False,
) -> Text:
    """A coloured block bar showing ``used`` out of ``total``.

    When ``invert`` is true the colour reflects *availability* (free) rather
    than utilisation, which reads better for "free GPU" tables.
    """
    bar = Text()
    if total <= 0:
        bar.append("─" * width, style="dim")
        return bar
    used = max(0, min(used, total))
    filled = round(used / total * width)
    pct = used / total * 100
    if invert:
        style = availability_style(total - used, total)
    else:
        style = util_style(pct)
    bar.append("█" * filled, style=style)
    bar.append("░" * (width - filled), style="grey37")
    return bar


def percent_bar(pct: float, width: int = 18) -> Text:
    """A bar for an elapsed/completion percentage (0-100)."""
    pct = max(0.0, min(pct, 100.0))
    filled = round(pct / 100 * width)
    # For time elapsed, "almost out of time" should look urgent.
    if pct >= 90:
        style = "bright_red"
    elif pct >= 75:
        style = "yellow"
    else:
        style = "bright_green"
    bar = Text()
    bar.append("█" * filled, style=style)
    bar.append("░" * (width - filled), style="grey37")
    return bar


def gpu_chiplets(types: dict[str, int]) -> Text:
    """Render ``{type: n}`` as coloured ``n×type`` chips."""
    text = Text()
    first = True
    for gpu_type, count in sorted(types.items()):
        if not first:
            text.append("  ")
        first = False
        text.append(f"{count}×", style="bold")
        text.append(gpu_type, style=gpu_color(gpu_type))
    if first:
        text.append("-", style="dim")
    return text
