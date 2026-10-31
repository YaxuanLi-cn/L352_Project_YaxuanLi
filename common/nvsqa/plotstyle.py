import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GRAY = "#b5b4ad"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fdf4ef", "#f9d3c2", "#f3a685", "#eb6834", "#b84a1f", "#7d3113"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#9c2423"])
METHOD_COLOR = {"3dgs": BLUE, "nerf": ORANGE}
METHOD_LABEL = {"3dgs": "3DGS", "nerf": "NeRF"}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.6,
        "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def save(fig, path_without_ext, formats=("pdf", "png")):
    for ext in formats:
        fig.savefig(f"{path_without_ext}.{ext}")
    plt.close(fig)
