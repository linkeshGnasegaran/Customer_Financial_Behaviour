from io import BytesIO


def get_chart_colors(theme):
    if theme == "Dark Mode":
        return {
            "bg": "#020617",
            "text": "#f8fafc",
            "grid": "#64748b",
            "bar": "#38bdf8",
            "scatter": "#22c55e",
            "centroid": "#facc15"
        }

    return {
        "bg": "#ffffff",
        "text": "#0f172a",
        "grid": "#cbd5e1",
        "bar": "#2563eb",
        "scatter": "#16a34a",
        "centroid": "#dc2626"
    }


def style_chart(fig, ax, title, theme):
    colors = get_chart_colors(theme)

    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])

    ax.set_title(title, color=colors["text"], fontsize=14, fontweight="bold")
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.tick_params(axis="x", colors=colors["text"])
    ax.tick_params(axis="y", colors=colors["text"])

    for spine in ax.spines.values():
        spine.set_color(colors["grid"])

    ax.grid(True, alpha=0.25, color=colors["grid"])
    fig.tight_layout()

    return colors


def fig_to_png(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer