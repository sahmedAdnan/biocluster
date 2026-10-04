"""
app/theme.py
------------
Design system for Bio Cluster: tokens, palettes and the application stylesheet.

Everything visual comes from here. Widgets never carry their own hex strings, so
the palette cannot drift the way two unrelated greens once did, and switching to
dark mode is a matter of swapping a dict rather than hunting inline stylesheets.

Structure
---------
    PALETTES        light + dark token sets
    set_theme(name) rebinds the module-level tokens
    qss()           the full QApplication stylesheet, built from the active tokens

Colour rules encoded here
-------------------------
1. ACTION COLOUR IS SEMANTIC. Three roles -- primary, danger, neutral -- plus one
   reserved accent for annotation. Only the primary action is filled, so the eye
   lands where it should without spending more colour.

2. EVERY FILLED BUTTON PASSES WCAG AA (4.5:1) FOR ITS LABEL. The old primary
   green (#2ed573) sat at 1.93:1 -- the most-clicked control in the app had an
   all-but-illegible label. Measured ratios are noted per token.

3. STATUS COLOUR IS RESERVED. good / warning / critical mean exactly that and are
   never reused as chart series colours.

4. THE CATEGORICAL CHART PALETTE IS CVD-SAFE AND NEVER CYCLES. Okabe-Ito,
   validated for deuteranopia and tritanopia separation. Past CATEGORICAL_MAX
   groups, colour stops being an identity key -- fall back to NEUTRAL_MARK plus
   interaction rather than recycling hues.
"""

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
# Neutrals carry a slight cool bias rather than being pure grey, so they sit
# with the green primary instead of fighting it.

_LIGHT = dict(
    # surfaces
    WINDOW        = "#eef1f3",   # app ground, behind everything
    SURFACE       = "#ffffff",   # cards, panels
    SURFACE_SUNK  = "#f5f7f8",   # inset wells, table stripes
    SURFACE_RAIL  = "#12181c",   # the icon rail is always dark, in both themes
    BORDER        = "#dde3e7",   # hairlines
    BORDER_STRONG = "#c3ccd3",   # input outlines
    OVERLAY       = "rgba(0,0,0,0.06)",

    # ink
    INK           = "#141a1d",
    INK_SECONDARY = "#47535a",
    INK_MUTED     = "#78868e",
    INK_ON_RAIL   = "#8b9aa3",
    INK_ON_ACTION = "#ffffff",

    # actions  (contrast = white text on the fill)
    PRIMARY       = "#15803d",   # 5.02:1
    PRIMARY_HOVER = "#166534",   # 7.13:1
    DANGER        = "#b91c1c",   # 6.47:1
    DANGER_HOVER  = "#991b1b",
    ACCENT        = "#1d4ed8",   # 6.70:1
    ACCENT_HOVER  = "#1e40af",
    SPECIAL       = "#7c3aed",   # 5.70:1
    SPECIAL_HOVER = "#6d28d9",

    # status -- reserved
    STATUS_GOOD     = "#15803d",
    STATUS_WARNING  = "#b45309",
    STATUS_CRITICAL = "#b91c1c",

    # charts
    GRID          = "#eceff2",
    NEUTRAL_MARK  = "#9aa4b0",
)

_DARK = dict(
    WINDOW        = "#0d1215",
    SURFACE       = "#161d21",
    SURFACE_SUNK  = "#111719",
    SURFACE_RAIL  = "#0a0e11",
    BORDER        = "#28333a",
    BORDER_STRONG = "#3a4750",
    OVERLAY       = "rgba(255,255,255,0.07)",

    INK           = "#eef3f5",
    INK_SECONDARY = "#a9b7bf",
    INK_MUTED     = "#7d8b94",
    INK_ON_RAIL   = "#7f8e97",
    INK_ON_ACTION = "#06120a",

    # Lifted for legibility on a dark ground; the dark theme is chosen, not
    # an automatic inversion of the light one.
    PRIMARY       = "#3fbf6a",
    PRIMARY_HOVER = "#57d07f",
    DANGER        = "#f26d6d",
    DANGER_HOVER  = "#ff8585",
    ACCENT        = "#6f9dff",
    ACCENT_HOVER  = "#8bb0ff",
    SPECIAL       = "#a982f7",
    SPECIAL_HOVER = "#bb9bff",

    STATUS_GOOD     = "#3fbf6a",
    STATUS_WARNING  = "#e0a33c",
    STATUS_CRITICAL = "#f26d6d",

    GRID          = "#222c32",
    NEUTRAL_MARK  = "#5c6a73",
)

PALETTES = {"light": _LIGHT, "dark": _DARK}
ACTIVE = "light"


# ---------------------------------------------------------------------------
# Spacing, radius, type  (theme-independent)
# ---------------------------------------------------------------------------
SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG, SPACE_XL = 4, 8, 14, 22, 34
RADIUS_SM, RADIUS_MD, RADIUS_LG = 5, 9, 14

FONT_FAMILY   = "Segoe UI"
FONT_MONO     = "Consolas"
TYPE_HERO     = 30    # stat tile value
TYPE_H1       = 19    # page title
TYPE_H2       = 15    # card title
TYPE_BODY     = 13
TYPE_SMALL    = 12
TYPE_CAPTION  = 11
TYPE_LABEL    = 10    # uppercase section eyebrows
# Kept because main.py refers to it; the H1/H2 names came later.
TYPE_TITLE    = TYPE_H1
TYPE_SUBTITLE = TYPE_BODY

RAIL_W        = 68
PANEL_W       = 330

# ---------------------------------------------------------------------------
# Control sizing
# ---------------------------------------------------------------------------
# One height for every ordinary control, one for the two page-level CTAs.
# Buttons used to carry ad-hoc setMinimumHeight() calls (35, 38, 45) plus
# inline padding, so a single toolbar held three different button heights.
# Height is a token now; call sites must not set it.
CONTROL_H     = 32    # toolbar / inline buttons, combos, spin boxes
CONTROL_H_LG  = 42    # "Run Pipeline", "Annotate Clusters" -- the CTAs only

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
# Okabe-Ito, fixed order. Assign by entity, never by rank, never cycled.
CATEGORICAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
CATEGORICAL_MAX = len(CATEGORICAL)
SEQUENTIAL_HUE  = "#0072B2"


def rgba(hex_color: str, alpha: float) -> str:
    """Qt stylesheets do NOT support #RRGGBBAA -- an 8-digit hex is silently
    misparsed (it rendered as yellow). Always express translucency as rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def set_theme(name: str = "light") -> None:
    """Rebind the module-level colour tokens to the named palette."""
    global ACTIVE
    ACTIVE = name if name in PALETTES else "light"
    globals().update(PALETTES[ACTIVE])


set_theme("light")          # populate the tokens at import time


def is_dark() -> bool:
    return ACTIVE == "dark"


# ---------------------------------------------------------------------------
# Small helpers kept for call sites that build inline styles
# ---------------------------------------------------------------------------
def button_qss(role: str = "primary") -> str:
    """Explicit style for one action button.

    Prefer setProperty('role', ...) + a re-polish: an inline stylesheet
    replaces the global QPushButton rule wholesale, and Qt then measures
    min-height against a different box, so an inline-styled button lands a
    couple of pixels off from the themed buttons beside it. Kept for call
    sites outside main.py; the sizes below mirror the global sheet."""
    fills = {"primary": (PRIMARY, PRIMARY_HOVER),
             "danger":  (DANGER,  DANGER_HOVER),
             "special": (SPECIAL, SPECIAL_HOVER),
             "accent":  (ACCENT,  ACCENT_HOVER)}
    # primary/danger/special are the page-level CTAs (Run Pipeline, Stop,
    # Annotate Clusters) and share the taller height.
    h = CONTROL_H_LG if role in ("primary", "danger", "special") else CONTROL_H
    if role == "neutral":
        return (f"QPushButton{{background:transparent;color:{ACCENT};"
                f"border:1px solid {BORDER_STRONG};border-radius:{RADIUS_SM}px;"
                f"padding:0 14px;font-weight:600;min-height:{h}px;}}"
                f"QPushButton:hover{{background:{SURFACE_SUNK};border-color:{ACCENT};}}"
                f"QPushButton:disabled{{color:{INK_MUTED};border-color:{BORDER};}}")
    base, hover = fills.get(role, fills["primary"])
    return (f"QPushButton{{background:{base};color:{INK_ON_ACTION};border:none;"
            f"border-radius:{RADIUS_SM}px;padding:0 14px;font-weight:600;"
            f"min-height:{h}px;}}"
            f"QPushButton:hover{{background:{hover};}}"
            f"QPushButton:disabled{{background:{BORDER};color:{INK_MUTED};}}")


def caption_qss(color: str = None) -> str:
    return f"font-size:{TYPE_CAPTION}px;color:{color or INK_MUTED};"


def card_qss(accent: str = None) -> str:
    left = f"border-left:4px solid {accent};" if accent else ""
    return (f"QFrame{{background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:{RADIUS_MD}px;{left}}}")


def status_for(metric: str, value) -> str:
    """Map a metric value to a reserved status colour, so a card's colour means
    something rather than decorating. None when there is no honest threshold."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if metric in {"ARI", "NMI", "Hungarian", "Hungarian Accuracy"}:
        return STATUS_GOOD if v >= 0.80 else (STATUS_WARNING if v >= 0.50 else STATUS_CRITICAL)
    if metric == "Silhouette":
        return STATUS_GOOD if v >= 0.50 else (STATUS_WARNING if v >= 0.25 else STATUS_CRITICAL)
    # Calinski-Harabasz and Davies-Bouldin are unbounded: no absolute "good".
    return None


# ---------------------------------------------------------------------------
# Application stylesheet
# ---------------------------------------------------------------------------
def qss() -> str:
    """Full stylesheet for the app, generated from the active palette."""
    SEL_TINT = rgba(ACCENT, 0.20 if is_dark() else 0.16)
    return f"""
/* Only colour here. Setting font-size on * would beat every setFont() call
   and silently override deliberate per-widget sizing -- the base family and
   size are applied once via QApplication.setFont() instead. */
* {{ color: {INK}; }}
QMainWindow, QDialog {{ background: {WINDOW}; }}
QWidget#Canvas   {{ background: {WINDOW}; }}
QWidget#Panel    {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
QWidget#Rail     {{ background: {SURFACE_RAIL}; }}

/* ---------- rail ---------- */
QToolButton#RailBtn {{
    background: transparent; border: none; border-radius: {RADIUS_MD}px;
    color: {INK_ON_RAIL}; padding: 9px 0; font-size: {TYPE_LABEL}px;
}}
QToolButton#RailBtn:hover  {{ background: rgba(255,255,255,0.08); color: #dfe7ec; }}
QToolButton#RailBtn:checked{{ background: rgba(255,255,255,0.14); color: #ffffff; }}

/* ---------- cards & sections ---------- */
QFrame#Card {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: {RADIUS_MD}px;
}}
QLabel#PageTitle  {{ font-size: {TYPE_H1}px; font-weight: 700; color: {INK}; }}
QLabel#CardTitle  {{ font-size: {TYPE_H2}px; font-weight: 650; color: {INK}; }}
QLabel#Eyebrow    {{
    font-size: {TYPE_LABEL}px; font-weight: 700; color: {INK_MUTED};
    letter-spacing: 1.2px;
}}
QLabel#Caption    {{ font-size: {TYPE_CAPTION}px; color: {INK_MUTED}; }}
QLabel#Muted      {{ color: {INK_SECONDARY}; }}
QLabel#StatValue  {{ font-size: {TYPE_HERO}px; font-weight: 700; color: {INK}; }}
QLabel#StatLabel  {{ font-size: {TYPE_CAPTION}px; font-weight: 600; color: {INK_MUTED}; }}
QLabel#EmptyGlyph {{ font-size: 46px; color: {BORDER_STRONG}; }}

/* ---------- inputs ---------- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {SURFACE}; border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_SM}px; padding: 0 9px; min-height: {CONTROL_H}px;
    selection-background-color: {ACCENT}; selection-color: {INK_ON_ACTION};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border-color: {INK_MUTED};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT}; selection-color: {INK_ON_ACTION};
    outline: none;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{ width: 15px; border: none; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {BORDER_STRONG}; background: {SURFACE};
}}
QCheckBox::indicator:checked {{ background: {PRIMARY}; border-color: {PRIMARY}; }}

/* ---------- buttons ---------- */
QPushButton {{
    background: transparent; color: {INK}; border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_SM}px; padding: 0 14px; font-weight: 600;
    min-height: {CONTROL_H}px; font-size: {TYPE_SMALL}px;
}}
QPushButton:hover    {{ background: {SURFACE_SUNK}; border-color: {INK_MUTED}; }}
QPushButton:disabled {{ color: {INK_MUTED}; border-color: {BORDER}; }}
QPushButton[role="primary"] {{
    background: {PRIMARY}; color: {INK_ON_ACTION}; border: 1px solid {PRIMARY};
    padding: 0 16px; font-size: {TYPE_BODY}px; min-height: {CONTROL_H_LG}px;
}}
QPushButton[role="primary"]:hover    {{ background: {PRIMARY_HOVER}; }}
QPushButton[role="primary"]:disabled {{ background: {BORDER}; color: {INK_MUTED}; }}
QPushButton[role="danger"] {{
    background: {DANGER}; color: #ffffff; border: 1px solid {DANGER}; padding: 0 16px;
    font-size: {TYPE_BODY}px; min-height: {CONTROL_H_LG}px;
}}
QPushButton[role="danger"]:hover  {{ background: {DANGER_HOVER}; }}
/* Accent: a filled action that is NOT the page CTA -- report generation,
   "apply these settings", "show more". Toolbar height, so it lines up with
   the outline buttons beside it instead of standing a few pixels taller. */
/* A filled role keeps a same-colour 1px border rather than border:none, so
   its box is measured exactly like the outline buttons it sits beside --
   otherwise a filled button lands 2px shorter in the same toolbar. */
QPushButton[role="accent"] {{
    background: {ACCENT}; color: #ffffff; border: 1px solid {ACCENT};
}}
QPushButton[role="accent"]:hover    {{ background: {ACCENT_HOVER}; }}
QPushButton[role="accent"]:disabled {{ background: {BORDER}; color: {INK_MUTED}; }}
QPushButton[role="special"] {{
    background: {SPECIAL}; color: #ffffff; border: 1px solid {SPECIAL}; padding: 0 16px;
    font-size: {TYPE_BODY}px; min-height: {CONTROL_H_LG}px;
}}
QPushButton[role="special"]:hover    {{ background: {SPECIAL_HOVER}; }}
QPushButton[role="special"]:disabled {{ background: {BORDER}; color: {INK_MUTED}; }}
/* Neutral: an outline button that reads as secondary. Same box as the plain
   QPushButton above -- only the label colour differs -- so it lines up with
   its neighbours instead of standing 2px short. */
QPushButton[role="neutral"] {{
    background: transparent; color: {ACCENT};
    border: 1px solid {BORDER_STRONG};
}}
QPushButton[role="neutral"]:hover    {{ background: {SURFACE_SUNK}; border-color: {ACCENT}; }}
QPushButton[role="neutral"]:disabled {{ color: {INK_MUTED}; border-color: {BORDER}; }}
QPushButton[role="ghost"] {{
    border: none; color: {ACCENT}; padding: 6px 8px;
}}
QPushButton[role="ghost"]:hover {{ background: {OVERLAY}; }}

/* ---------- segmented control ---------- */
QPushButton[seg="true"] {{
    border: 1px solid {BORDER}; background: {SURFACE_SUNK};
    color: {INK_SECONDARY}; padding: 0 16px; font-weight: 600;
    border-radius: {RADIUS_SM}px; min-height: {CONTROL_H}px;
}}
QPushButton[seg="true"]:checked {{
    background: {SURFACE}; color: {INK}; border-color: {BORDER_STRONG};
}}

/* ---------- tables ---------- */
QTableWidget, QTableView {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: {RADIUS_MD}px; gridline-color: {BORDER};
    selection-background-color: {ACCENT}; selection-color: {INK_ON_ACTION};
    alternate-background-color: {SURFACE_SUNK};
}}
QHeaderView::section {{
    background: {SURFACE_SUNK}; color: {INK_MUTED}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 8px 10px;
    font-size: {TYPE_CAPTION}px; font-weight: 700;
}}
QTableWidget::item {{ padding: 6px 8px; }}
/* Selection: background and text colour must be set together. qdarktheme
   paints the selected row a pale tint via its stylesheet while the palette
   still supplies white HighlightedText -- white on #d7e0ea, ~1.3:1, which
   made the default-selected top row unreadable. A light accent tint with
   normal ink keeps the row legible and stays quieter than a filled band. */
QTableWidget::item:selected, QTableView::item:selected {{
    background: {SEL_TINT}; color: {INK};
}}
QTableWidget::item:selected:!active, QTableView::item:selected:!active {{
    background: {SEL_TINT}; color: {INK};
}}

/* ---------- misc ---------- */
QProgressBar {{
    background: {SURFACE_SUNK}; border: none; border-radius: 4px;
    height: 7px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {PRIMARY}; border-radius: 4px; }}
/* A QScrollArea paints its viewport with the default window brush, which
   showed as a grey slab inside the white panel. Both the area and the
   viewport's child widget have to be made transparent. */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG}; border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {INK_MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; }}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG}; border-radius: 5px; min-width: 28px;
}}
QToolTip {{
    background: {INK}; color: {WINDOW}; border: none;
    padding: 7px 10px; border-radius: {RADIUS_SM}px;
}}
QSplitter::handle {{ background: {BORDER}; width: 1px; }}
"""
