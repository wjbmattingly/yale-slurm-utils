"""Colours, styles and small visual helpers, driven by a swappable theme.

A :class:`Theme` is a flat set of colour "roles". Built-in themes live in
``BUILTIN_THEMES``; users can drop their own YAML themes in the config themes
directory (see :func:`~yale_slurm_utils.config.themes_dir`). The active theme is
a module-global so the (functional) render helpers can stay simple.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from rich.text import Text

from . import config

# --------------------------------------------------------------------------- #
# Theme model
# --------------------------------------------------------------------------- #
_DEFAULT_PALETTE = [
    "bright_green",
    "bright_magenta",
    "bright_cyan",
    "bright_yellow",
    "bright_blue",
    "bright_red",
    "green",
    "magenta",
]


@dataclass
class Theme:
    """A flat palette of colour roles. All fields have dark-theme defaults, so
    a user YAML theme only needs to override what it wants to change."""

    name: str = "custom"
    mode: str = "dark"  # "dark" or "light"
    # branding / structure
    primary: str = "#00356b"
    accent: str = "#286dc0"
    heading: str = "white"
    border: str = "grey37"
    zebra: str = "grey11"
    selection: str = "grey30"
    bar_empty: str = "grey37"
    # node / job states
    idle: str = "green"
    mixed: str = "yellow"
    allocated: str = "bright_red"
    down: str = "red dim"
    pending: str = "bright_yellow"
    completed: str = "cyan"
    # utilisation heat (low -> high load)
    util_cool: str = "bright_green"
    util_ok: str = "green"
    util_warm: str = "yellow"
    util_hot: str = "bright_red"
    # availability of free resources (none / scarce / plenty)
    avail_none: str = "bright_red"
    avail_low: str = "yellow"
    avail_plenty: str = "bright_green"
    # per-GPU-type palette (cycled)
    gpu_palette: list[str] = field(default_factory=lambda: list(_DEFAULT_PALETTE))

    def to_yaml_dict(self) -> dict:
        return asdict(self)


_FIELD_NAMES = {f.name for f in fields(Theme)}


def theme_from_dict(data: dict, name: str | None = None) -> Theme:
    """Build a :class:`Theme` from a (possibly partial) mapping.

    Unknown keys are ignored; missing keys fall back to the dark defaults.
    """
    kwargs = {k: v for k, v in (data or {}).items() if k in _FIELD_NAMES}
    if name is not None:
        kwargs["name"] = name
    return Theme(**kwargs)


# --------------------------------------------------------------------------- #
# Built-in themes
# --------------------------------------------------------------------------- #
BUILTIN_THEMES: dict[str, Theme] = {
    "dark": Theme(name="dark", mode="dark"),
    "light": Theme(
        name="light",
        mode="light",
        primary="#00356b",
        accent="#0f4d92",
        heading="grey15",
        border="grey54",
        zebra="grey85",
        selection="grey70",
        bar_empty="grey74",
        idle="green4",
        mixed="dark_orange3",
        allocated="red3",
        down="grey42",
        pending="dark_orange3",
        completed="dark_cyan",
        util_cool="green4",
        util_ok="yellow4",
        util_warm="dark_orange3",
        util_hot="red3",
        avail_none="red3",
        avail_low="dark_orange3",
        avail_plenty="green4",
        gpu_palette=[
            "green4", "magenta3", "dark_cyan", "yellow4",
            "blue3", "red3", "purple4", "deep_pink4",
        ],
    ),
    "solarized-dark": Theme(
        name="solarized-dark", mode="dark",
        primary="#268bd2", accent="#2aa198", heading="#eee8d5",
        border="#586e75", zebra="#073642", selection="#0a4b53", bar_empty="#586e75",
        idle="#859900", mixed="#b58900", allocated="#dc322f", down="#586e75",
        pending="#cb4b16", completed="#2aa198",
        util_cool="#859900", util_ok="#859900", util_warm="#b58900", util_hot="#dc322f",
        avail_none="#dc322f", avail_low="#b58900", avail_plenty="#859900",
        gpu_palette=[
            "#859900", "#d33682", "#2aa198", "#b58900",
            "#268bd2", "#dc322f", "#6c71c4", "#cb4b16",
        ],
    ),
    "solarized-light": Theme(
        name="solarized-light", mode="light",
        primary="#268bd2", accent="#268bd2", heading="#586e75",
        border="#93a1a1", zebra="#eee8d5", selection="#d8d0b8", bar_empty="#93a1a1",
        idle="#859900", mixed="#b58900", allocated="#dc322f", down="#93a1a1",
        pending="#cb4b16", completed="#2aa198",
        util_cool="#859900", util_ok="#859900", util_warm="#b58900", util_hot="#dc322f",
        avail_none="#dc322f", avail_low="#b58900", avail_plenty="#859900",
        gpu_palette=[
            "#859900", "#d33682", "#2aa198", "#b58900",
            "#268bd2", "#dc322f", "#6c71c4", "#cb4b16",
        ],
    ),
    "dracula": Theme(
        name="dracula", mode="dark",
        primary="#bd93f9", accent="#ff79c6", heading="#f8f8f2",
        border="#6272a4", zebra="#343746", selection="#44475a", bar_empty="#6272a4",
        idle="#50fa7b", mixed="#f1fa8c", allocated="#ff5555", down="#6272a4",
        pending="#ffb86c", completed="#8be9fd",
        util_cool="#50fa7b", util_ok="#50fa7b", util_warm="#f1fa8c", util_hot="#ff5555",
        avail_none="#ff5555", avail_low="#ffb86c", avail_plenty="#50fa7b",
        gpu_palette=[
            "#50fa7b", "#ff79c6", "#8be9fd", "#f1fa8c",
            "#bd93f9", "#ff5555", "#ffb86c", "#8be9fd",
        ],
    ),
    "nord": Theme(
        name="nord", mode="dark",
        primary="#88c0d0", accent="#81a1c1", heading="#eceff4",
        border="#4c566a", zebra="#3b4252", selection="#434c5e", bar_empty="#4c566a",
        idle="#a3be8c", mixed="#ebcb8b", allocated="#bf616a", down="#4c566a",
        pending="#d08770", completed="#8fbcbb",
        util_cool="#a3be8c", util_ok="#a3be8c", util_warm="#ebcb8b", util_hot="#bf616a",
        avail_none="#bf616a", avail_low="#ebcb8b", avail_plenty="#a3be8c",
        gpu_palette=[
            "#a3be8c", "#b48ead", "#88c0d0", "#ebcb8b",
            "#5e81ac", "#bf616a", "#d08770", "#8fbcbb",
        ],
    ),
    "gruvbox-dark": Theme(
        name="gruvbox-dark", mode="dark",
        primary="#83a598", accent="#fabd2f", heading="#ebdbb2",
        border="#665c54", zebra="#3c3836", selection="#504945", bar_empty="#665c54",
        idle="#b8bb26", mixed="#fabd2f", allocated="#fb4934", down="#665c54",
        pending="#fe8019", completed="#8ec07c",
        util_cool="#b8bb26", util_ok="#b8bb26", util_warm="#fabd2f", util_hot="#fb4934",
        avail_none="#fb4934", avail_low="#fabd2f", avail_plenty="#b8bb26",
        gpu_palette=[
            "#b8bb26", "#d3869b", "#8ec07c", "#fabd2f",
            "#83a598", "#fb4934", "#fe8019", "#d5c4a1",
        ],
    ),
}


# --------------------------------------------------------------------------- #
# Loading user themes / resolving the active theme
# --------------------------------------------------------------------------- #
def load_user_themes() -> dict[str, Theme]:
    """Load ``*.yaml`` / ``*.yml`` themes from the user's themes directory."""
    import yaml

    out: dict[str, Theme] = {}
    directory = config.themes_dir()
    for path in sorted(directory.glob("*.y*ml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        name = str(data.get("name") or path.stem)
        out[name] = theme_from_dict(data, name=name)
    return out


def all_themes() -> dict[str, Theme]:
    """Built-in themes plus user themes (user themes win on name clashes)."""
    themes = dict(BUILTIN_THEMES)
    themes.update(load_user_themes())
    return themes


def resolve_theme(name: str | None = None) -> Theme:
    """Pick a theme by ``name``; fall back through preference then auto-detect.

    ``name`` (or the saved preference) of ``"auto"``/``None`` triggers
    light/dark auto-detection of the terminal background.
    """
    themes = all_themes()
    requested = name or config.get_theme_preference() or "auto"
    if requested != "auto" and requested in themes:
        return themes[requested]
    if requested != "auto" and requested not in themes:
        # Unknown explicit name: fall back to auto rather than crashing.
        pass
    dark = config.detect_dark_background()
    return themes["dark"] if dark is not False else themes["light"]


# --------------------------------------------------------------------------- #
# Active theme + style helpers
# --------------------------------------------------------------------------- #
_ACTIVE: Theme = BUILTIN_THEMES["dark"]
_GPU_COLOR_CACHE: dict[str, str] = {}


def active() -> Theme:
    return _ACTIVE


def set_active(theme: Theme) -> None:
    global _ACTIVE, _GPU_COLOR_CACHE
    _ACTIVE = theme
    _GPU_COLOR_CACHE = {}


def apply(name: str | None = None) -> Theme:
    """Resolve and activate a theme in one call. Returns the active theme."""
    theme = resolve_theme(name)
    set_active(theme)
    return theme


# --- structural roles ------------------------------------------------------ #
def accent() -> str:
    return _ACTIVE.accent


def primary() -> str:
    return _ACTIVE.primary


def heading() -> str:
    return _ACTIVE.heading


def border() -> str:
    return _ACTIVE.border


def zebra_rows() -> list[str]:
    return ["", f"on {_ACTIVE.zebra}"]


def selection_style() -> str:
    return f"on {_ACTIVE.selection}"


# --- semantic colours ------------------------------------------------------ #
def gpu_color(gpu_type: str) -> str:
    palette = _ACTIVE.gpu_palette or _DEFAULT_PALETTE
    if gpu_type not in _GPU_COLOR_CACHE:
        idx = len(_GPU_COLOR_CACHE) % len(palette)
        _GPU_COLOR_CACHE[gpu_type] = palette[idx]
    return _GPU_COLOR_CACHE[gpu_type]


def state_style(state: str) -> str:
    s = state.lower()
    if "idle" in s:
        return _ACTIVE.idle
    if "mix" in s:
        return _ACTIVE.mixed
    if "alloc" in s or "running" in s:
        return _ACTIVE.allocated
    if "pending" in s:
        return _ACTIVE.pending
    if any(t in s for t in ("down", "fail", "drain", "maint")):
        return _ACTIVE.down
    if "complet" in s:
        return _ACTIVE.completed
    return _ACTIVE.heading


def util_style(pct: float) -> str:
    """Colour for a utilisation/occupancy percentage (busy == hot)."""
    if pct >= 90:
        return _ACTIVE.util_hot
    if pct >= 66:
        return _ACTIVE.util_warm
    if pct >= 33:
        return _ACTIVE.util_ok
    return _ACTIVE.util_cool


def availability_style(free: int, total: int) -> str:
    """Colour for *free* counts (more free == cooler/greener)."""
    if total == 0:
        return "dim"
    if free == 0:
        return _ACTIVE.avail_none
    if free / total <= 0.15:
        return _ACTIVE.avail_low
    return _ACTIVE.avail_plenty


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
    bar.append("░" * (width - filled), style=_ACTIVE.bar_empty)
    return bar


def percent_bar(pct: float, width: int = 18) -> Text:
    """A bar for an elapsed/completion percentage (0-100)."""
    pct = max(0.0, min(pct, 100.0))
    filled = round(pct / 100 * width)
    # For time elapsed, "almost out of time" should look urgent.
    if pct >= 90:
        style = _ACTIVE.util_hot
    elif pct >= 75:
        style = _ACTIVE.util_warm
    else:
        style = _ACTIVE.util_cool
    bar = Text()
    bar.append("█" * filled, style=style)
    bar.append("░" * (width - filled), style=_ACTIVE.bar_empty)
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
