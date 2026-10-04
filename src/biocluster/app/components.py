"""
app/components.py
-----------------
Reusable UI pieces for the Bio Cluster shell.

Each one is a plain QWidget subclass styled through object names that
app/theme.py's stylesheet targets, so nothing here carries its own colours and
a theme switch reaches every component for free.
"""

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from biocluster.app import theme as T


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def styled(w: QWidget, object_name: str) -> QWidget:
    """Give a plain QWidget an object name AND make it honour stylesheet
    backgrounds.

    Qt gotcha: a bare QWidget does not paint `background` from a stylesheet
    unless WA_StyledBackground is set -- QFrame does, QWidget does not. Without
    this the rail and panel silently render in the default window colour and the
    whole theme looks broken.
    """
    w.setObjectName(object_name)
    w.setAttribute(Qt.WA_StyledBackground, True)
    return w


def label(text, object_name=None, wrap=False, parent=None) -> QLabel:
    lb = QLabel(text, parent)
    if object_name:
        lb.setObjectName(object_name)
    lb.setWordWrap(wrap)
    return lb


def eyebrow(text) -> QLabel:
    """Small uppercase section marker. Used only where it encodes a real
    grouping, never as decoration."""
    return label(text.upper(), "Eyebrow")


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{T.BORDER};border:none;")
    return f


class IconRail(QWidget):
    """Slim vertical navigation. Emits the index of the selected destination.

    Glyphs are drawn from the font rather than image assets so the rail stays
    crisp at any DPI and needs nothing bundled by PyInstaller.
    """
    changed = Signal(int)

    def __init__(self, items, parent=None):
        super().__init__(parent)
        styled(self, "Rail")
        self.setFixedWidth(T.RAIL_W)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 14, 8, 14)
        lay.setSpacing(6)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for i, (glyph, text, tip) in enumerate(items):
            b = QToolButton()
            b.setObjectName("RailBtn")
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setText(f"{glyph}\n{text}")
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            # Size comes from the stylesheet (QToolButton#RailBtn); setting a
            # font here would win over it before the sheet is applied and the
            # labels would render oversized on first paint.
            self.group.addButton(b, i)
            lay.addWidget(b)
        lay.addStretch()
        self.group.idClicked.connect(self.changed.emit)

    def select(self, idx: int):
        btn = self.group.button(idx)
        if btn:
            btn.setChecked(True)


class Card(QFrame):
    """A titled surface. `body` is a layout you add your own widgets to."""

    def __init__(self, title=None, subtitle=None, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(T.SPACE_MD, T.SPACE_MD, T.SPACE_MD, T.SPACE_MD)
        outer.setSpacing(T.SPACE_SM)
        if title:
            outer.addWidget(label(title, "CardTitle"))
        if subtitle:
            outer.addWidget(label(subtitle, "Caption", wrap=True))
        self.body = QVBoxLayout()
        self.body.setSpacing(T.SPACE_SM)
        outer.addLayout(self.body)


class StatTile(QFrame):
    """One headline number.

    The accent stripe is derived from the VALUE via theme.status_for, so colour
    means good/poor rather than decorating. Metrics with no absolute threshold
    (Calinski-Harabasz, Davies-Bouldin) get no stripe at all.
    """

    def __init__(self, name, value, metric=None, raw=None, tooltip="", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        accent = T.status_for(metric, value if raw is None else raw) if metric else None
        if accent:
            self.setStyleSheet(
                f"QFrame#Card{{border-left:4px solid {accent};}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(T.SPACE_MD, T.SPACE_MD, T.SPACE_MD, T.SPACE_MD)
        lay.setSpacing(2)
        # Wrap rather than clip: metric names like "ADJUSTED RAND INDEX" are
        # wider than a comfortable tile, and a truncated label is useless.
        n = label(name, "StatLabel", wrap=True)
        n.setMinimumHeight(26)
        n.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(n)
        lay.addWidget(label(str(value), "StatValue"))
        lay.addStretch()
        if tooltip:
            self.setToolTip(tooltip)
        self.setMinimumWidth(168)


# ---------------------------------------------------------------------------
# Collapsible settings section
# ---------------------------------------------------------------------------
class Section(QWidget):
    """A settings group that collapses to a one-line summary.

    The old UI kept seven of these expanded at once, which forced a scroll and
    made the panel read as a wall. Collapsed sections keep their summary visible
    so nothing is hidden, only quietened.
    """

    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._btn = QToolButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.setCursor(Qt.PointingHandCursor)
        self._btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn.setStyleSheet(
            f"QToolButton{{border:none;background:transparent;padding:7px 0;"
            f"font-size:{T.TYPE_LABEL}px;font-weight:700;color:{T.INK_MUTED};"
            f"letter-spacing:1.2px;text-align:left;}}"
            f"QToolButton:hover{{color:{T.INK};}}")
        self._title = title
        lay.addWidget(self._btn)

        self._summary = label("", "Caption")
        self._summary.setContentsMargins(0, 0, 0, T.SPACE_SM)
        lay.addWidget(self._summary)

        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 0, 0, T.SPACE_MD)
        self.body.setSpacing(T.SPACE_SM)
        lay.addWidget(self.content)

        self._btn.toggled.connect(self._on_toggle)
        self._on_toggle(expanded)

    def _on_toggle(self, on):
        self.content.setVisible(on)
        self._summary.setVisible(not on and bool(self._summary.text()))
        self._btn.setText(("▾  " if on else "▸  ") + self._title.upper())

    def set_summary(self, text):
        """One-line description shown while collapsed, e.g. 'UMAP · 20d · cosine'."""
        self._summary.setText(text)
        self._summary.setVisible(not self._btn.isChecked() and bool(text))


class EmptyState(QWidget):
    """Shown where results will appear, before there are any.

    A blank panel on launch reads as broken; a prompt that names the next action
    reads as ready. This is the cheapest perceived-quality win in the app.
    """

    def __init__(self, glyph, title, body, action=None, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        outer.setSpacing(T.SPACE_SM)

        g = label(glyph, "EmptyGlyph")
        g.setAlignment(Qt.AlignCenter)
        outer.addWidget(g)

        t = label(title, "CardTitle")
        t.setAlignment(Qt.AlignCenter)
        outer.addWidget(t)

        b = label(body, "Caption", wrap=True)
        b.setAlignment(Qt.AlignCenter)
        b.setMaximumWidth(430)
        outer.addWidget(b, alignment=Qt.AlignCenter)

        if action:
            text, slot = action
            btn = QPushButton(text)
            btn.setProperty("role", "primary")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(slot)
            btn.setFixedWidth(190)
            outer.addSpacing(T.SPACE_SM)
            outer.addWidget(btn, alignment=Qt.AlignCenter)


class Segmented(QWidget):
    """Horizontal exclusive selector, used to switch result views without
    spending a whole tab bar on them."""
    changed = Signal(int)

    def __init__(self, options, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        n = len(options)
        for i, text in enumerate(options):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setProperty("seg", "true")
            b.setCursor(Qt.PointingHandCursor)
            # round only the outer corners so the group reads as one control
            r = T.RADIUS_SM
            if i == 0:
                b.setStyleSheet(f"border-top-left-radius:{r}px;border-bottom-left-radius:{r}px;")
            if i == n - 1:
                b.setStyleSheet(b.styleSheet() +
                                f"border-top-right-radius:{r}px;border-bottom-right-radius:{r}px;")
            self.group.addButton(b, i)
            lay.addWidget(b)
        lay.addStretch()
        self.group.idClicked.connect(self.changed.emit)
        if options:
            self.group.button(0).setChecked(True)

    def select(self, i):
        b = self.group.button(i)
        if b:
            b.setChecked(True)


class Pill(QLabel):
    """Small status chip. State is encoded in shape and text as well as colour,
    so it survives greyscale and colour-vision deficiency."""

    def __init__(self, text, kind="neutral", parent=None):
        super().__init__(text, parent)
        colors = {"good": T.STATUS_GOOD, "warn": T.STATUS_WARNING,
                  "bad": T.STATUS_CRITICAL, "neutral": T.INK_MUTED}
        c = colors.get(kind, T.INK_MUTED)
        self.setStyleSheet(
            f"background:{T.rgba(c, 0.13)};color:{c};"
            f"border:1px solid {T.rgba(c, 0.34)};"
            f"border-radius:9px;padding:3px 10px;font-size:{T.TYPE_CAPTION}px;"
            f"font-weight:700;")
        self.setAlignment(Qt.AlignCenter)

# ---------------------------------------------------------------------------
# FlowLayout
# ---------------------------------------------------------------------------
class FlowLayout(QLayout):
    """A horizontal layout that wraps to the next line instead of squeezing.

    QHBoxLayout has no answer when the sum of its children's minimum widths
    exceeds the space available: it shrinks them past their minimums and the
    widgets visually collide. The gallery toolbar needed ~1110px of minimum
    width, so on anything narrower than a ~1500px window the cluster combo
    rode over the "Select Cluster to Verify:" label.

    This lays the same children out row by row, starting a new row whenever
    the next item would not fit, and reports the height that wrapping needs
    via heightForWidth(). Nothing is ever compressed below its size hint, so
    nothing can overlap at any window width.
    """

    def __init__(self, parent=None, margin=0, h_spacing=8, v_spacing=8):
        super().__init__(parent)
        self._items = []
        self._h = h_spacing
        self._v = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    # -- QLayout plumbing ---------------------------------------------------
    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    # -- wrapping -----------------------------------------------------------
    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        w = h = 0
        for it in self._items:
            s = it.minimumSize()
            w = max(w, s.width())
            h = max(h, s.height())
        m = self.contentsMargins()
        # Width is one item, not the sum: a single column is a legal layout,
        # which is exactly what lets the row wrap rather than force the
        # window wider.
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def _layout(self, rect, test_only):
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = eff.x(), eff.y(), 0

        for it in self._items:
            hint = it.sizeHint()
            nx = x + hint.width()
            if row_h > 0 and nx > eff.right() + 1:
                x = eff.x()
                y = y + row_h + self._v
                nx = x + hint.width()
                row_h = 0
            if not test_only:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x = nx + self._h
            row_h = max(row_h, hint.height())

        return y + row_h - rect.y() + m.bottom()


class FlowBar(QWidget):
    """A widget wrapping a FlowLayout, so it can be dropped straight into a
    QVBoxLayout with addWidget() and report its wrapped height correctly."""

    def __init__(self, parent=None, h_spacing=8, v_spacing=8):
        super().__init__(parent)
        self._flow = FlowLayout(self, margin=0, h_spacing=h_spacing, v_spacing=v_spacing)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def addWidget(self, w):
        self._flow.addWidget(w)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._flow.heightForWidth(w)

    def sizeHint(self):
        return QSize(self._flow.sizeHint().width(),
                     self._flow.heightForWidth(max(self.width(), 1)))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.setMinimumHeight(self._flow.heightForWidth(self.width()))
