"""
Terminal formatting utilities for clean, readable CLI output.

All functions write to stdout. They use only ASCII characters so they
work correctly on Windows cmd, PowerShell, and any Unix terminal without
requiring colorama or other dependencies.
"""

import os

# Column width for the metrics table
_COL_W = 14


# -------------------------------------------------------
# Section headers
# -------------------------------------------------------

def header(title: str, width: int = 68) -> None:
    """Print a bold section header."""
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def subheader(title: str, width: int = 68) -> None:
    """Print a lighter sub-section divider."""
    print(f"\n  {'-' * (width - 4)}")
    print(f"  {title}")
    print(f"  {'-' * (width - 4)}")


def step(number: int, total: int, description: str) -> None:
    """Print a step indicator."""
    print(f"\n  [{number}/{total}]  {description}")
    print(f"         {'.' * 50}")


def metrics_table(rows: list, title: str = "", width: int = 68) -> None:
    """
    Print a formatted metrics table.

    Args:
        rows:  List of dicts, each with 'label' (str) and 'value' (str or float).
               Optionally include 'note' for a right-hand annotation.
        title: Optional header above the table.
        width: Total table width.

    Example:
        metrics_table([
            {"label": "ARI",      "value": 0.9233},
            {"label": "NMI",      "value": 0.9897},
            {"label": "Accuracy", "value": 0.9359},
            {"label": "Clusters", "value": 186,    "note": "/ 200 expected"},
            {"label": "Noise",    "value": "5.3%"},
        ], title="DBSCAN Results")
    """
    if title:
        print(f"\n  {title}")
        print(f"  {'-' * (width - 4)}")

    for row in rows:
        label = row.get("label", "")
        val   = row.get("value", "")
        note  = row.get("note",  "")

        # Format numeric values
        if isinstance(val, float):
            val_str = f"{val:.4f}"
        elif isinstance(val, int):
            val_str = f"{val:,}"
        else:
            val_str = str(val)

        note_str = f"  {note}" if note else ""
        print(f"    {label:<24s}  {val_str:>10s}{note_str}")

    print(f"  {'-' * (width - 4)}")


def comparison_table(methods: list, metrics: dict,
                     title: str = "Method Comparison",
                     width: int = 72) -> None:
    """
    Print a multi-column comparison table.

    Args:
        methods:  Ordered list of method name strings.
        metrics:  Dict of {metric_label: {method_name: value}}.
        title:    Table title.
        width:    Total width.

    Example:
        comparison_table(
            methods=["kmeans", "dbscan", "hdbscan"],
            metrics={
                "ARI":      {"kmeans": 0.91, "dbscan": 0.92, "hdbscan": 0.97},
                "Accuracy": {"kmeans": 0.88, "dbscan": 0.94, "hdbscan": 0.97},
            }
        )
    """
    col_w = max(12, width // (len(methods) + 2))

    header_row = f"  {'Metric':<22s}" + "".join(
        f"{m[:col_w-2]:>{col_w}}" for m in methods
    )

    # Derive separator width from actual content, not the hint parameter
    actual_width = 22 + len(methods) * col_w

    print(f"\n  {title}")
    print(f"  {'=' * actual_width}")
    print(header_row)
    print(f"  {'-' * actual_width}")

    for metric_label, values in metrics.items():
        row = f"  {metric_label:<22s}"
        for method in methods:
            val = values.get(method, "-")
            if isinstance(val, float):
                cell = f"{val:.4f}"
            elif isinstance(val, int):
                cell = f"{val:,}"
            else:
                cell = str(val)
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"  {'=' * actual_width}")


def verdict(winner: str, reason: str) -> None:
    """Print a highlighted verdict line."""
    print(f"\n  >> Verdict: {winner}  ({reason})")


def warning(message: str) -> None:
    """Print a warning."""
    lines = message.strip().split("\n")
    print(f"\n  !! WARNING: {lines[0]}")
    for line in lines[1:]:
        print(f"             {line}")


def done(output_dir: str, elapsed: float = None) -> None:
    """Print a completion message."""
    parts = ["  Complete."]
    if elapsed is not None:
        mins  = int(elapsed // 60)
        secs  = elapsed % 60
        parts.append(f"Time: {mins}m {secs:.0f}s.")
    parts.append(f"Output: {os.path.abspath(output_dir)}")
    print("  " + "  ".join(parts))


def figure_saved(path: str) -> None:
    """Print a saved figure line, consistently formatted."""
    print(f"    [fig]  {path}")


def saved(label: str, path: str) -> None:
    """Print a saved file line."""
    print(f"    [save] {label:<16s}  {path}")
