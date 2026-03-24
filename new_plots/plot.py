import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
ANNOTATE_MODE = "best"  # "none", "best", "all"

sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

VENDOR_COLORS = {
    "CPU": "#005197",
    "NVIDIA": "#00D317",
    "AMD": "#971c00",
}

COLUMNS = [
    "year",
    "dp_tflops", # scalar/vector
    "sp_tflops", # scalar/vector
    "fp16_tflops", #tensor/matrix
    "fp8_tflops", #tensor/matrix
    "int8_tops",  #tensor matrix
    "cores",
    "mem_bw_GBs",
    "power_w",
    "transistors_b",
    "name",
    "abbr",
    "ref",
]


# ------------------------------------------------------------
# Data loader
# ------------------------------------------------------------
def load_file(path, vendor):
    df = pd.read_csv(path,
                     comment="#",
                     sep=r'\s+(?=(?:[^"]*"[^"]*")*[^"]*$)',
                     engine="python",
                     quotechar='"',
                     names=COLUMNS,
                     header=None)
    df["vendor"] = vendor
    return df


data = pd.concat([
    load_file("cpus.txt", "CPU"),
    load_file("nvidia_gpus.txt", "NVIDIA"),
    load_file("amd_gpus.txt", "AMD"),
],
                 ignore_index=True)

for col in ["year", "dp_tflops", "sp_tflops"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")


# ------------------------------------------------------------
# Plot function
# ------------------------------------------------------------
def plot_metric(metric, ylabel, title, outfile):
    plt.figure(figsize=(12, 8))
    texts = []
    x = []
    y = []
    first = True
    for vendor, g in data.groupby("vendor"):
        g = g.dropna(subset=[metric]).sort_values("year")
        if g.empty:
            continue

        color = VENDOR_COLORS.get(vendor, "black")




        plt.plot(g["year"], g[metric], marker="o", linewidth=3, label=vendor, color=color)

        # ----------------------------
        # Annotation selection
        # ----------------------------
        if ANNOTATE_MODE == "none":
            continue

        if ANNOTATE_MODE == "best":
            g = g.loc[g.groupby("year")[metric].idxmax()]

        # Annotation, use the abbreviation
        for _, r in g.iterrows():
            plt.text(r["year"],
                     r[metric] * 1.2,
                     r["abbr"].strip('\"'),
                     fontsize=10,
                     rotation=30,
                     color=color,
                     alpha=0.9)

        # Collission detection, never worked properly
        # for _, r in g.iterrows():
        #     x.append(r["year"])
        #     y.append(r[metric])
        #     texts.append(plt.text(r["year"], r[metric], r["abbr"], fontsize=10, rotation=30, color=color, alpha=0.9))

    # Collision-aware placement
    if texts:
         # never worked properly
        pass
        adjust_text(
            texts,
            x=x,
            y=y,
            autoalign='x',
            only_move={
                'points': '',
                'text': 'y'
            },
            force_points=0.15,
            # arrowprops=dict(arrowstyle="->", color='r', lw=0.5)
        )
     
    plt.yscale("log")
    # turn off scientific notation
    ax = plt.gca()

    # Major ticks
    ax.yaxis.set_major_locator(LogLocator(base=10))

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)

    # Minor ticks
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5]))
    ax.yaxis.set_minor_formatter(NullFormatter()) # add here formatter to show the ticks
    ax.minorticks_on()

    # Make ticks visible
    ax.tick_params(axis='y', which='minor', length=4, width=1)
    ax.tick_params(axis='y', which='major', length=7, width=1.2)
    
    ax.tick_params(axis='y', which='minor', left=True)

    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.savefig(outfile + ".pdf")
    print(ax.get_ylim())
    plt.close()


# ------------------------------------------------------------
# Generate plots
# ------------------------------------------------------------
plot_metric("dp_tflops", "Peak FP64 Performance [TFLOP/s]", "Peak Double Precision Performance Over Time",
            "dp_performance")

plot_metric("sp_tflops", "Peak FP32 Performance [TFLOP/s]", "Peak Single Precision Performance Over Time",
            "sp_performance")

print("!!!!!!!!!!!Note, the mem bw must be checked!!!!!!!!!!")
plot_metric("mem_bw_GBs", "Memory Bandwidth [GB/s]", "Memory Bandwidth Over Time",
            "mem_bw")
