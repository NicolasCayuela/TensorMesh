import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
def draw_error_bar(data, x,  y, hue, ax):
    _line_styles = ["-", "--", "-.", ":"]
    _markers = ["o", "s", "p", "^"]
    _colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    groups = data.groupby(hue)
    for i, (_hue, subdf) in enumerate(groups):
        subgroups = subdf.groupby(x)
        xs, means, stds = [], [], []
        for _x, subsubdf in subgroups:
            mean = subsubdf[y].mean()
            std  = subsubdf[y].std()
            xs.append(_x)
            means.append(mean)
            stds.append(std)
        ax.errorbar(xs, means, yerr=np.array(stds)*3, 
                    color=_colors[i], marker=_markers[i], linestyle=_line_styles[i], 
                    capsize=4.0, 
                    alpha=0.5,
                    # markersize=0.5,
                    label=_hue)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xlim(left=np.array(xs).min()/2, right=np.array(xs).max()*2)
    ax.legend()

def load_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0).to_dict()
    for key, value in df.items():
        df[key] = list(value.values())
    return df

if __name__ == '__main__':
    df2d = pd.read_csv("compare_linear_poisson_2d.csv")
    df3d = pd.read_csv("compare_linear_poisson_3d.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_error_bar(df2d, "degree of freedom", "time in s", "backend", axes[0])
    draw_error_bar(df3d, "degree of freedom", "time in s", "backend", axes[1])
    axes[0].set_title("(a)")
    axes[1].set_title("(b)")

    fig.savefig("compare_linear_poisson_time.png")
    fig.savefig("compare_linear_poisson_time.pdf")
    fig.savefig("compare_linear_poisson_time.eps")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    draw_error_bar(df2d, "degree of freedom", "CPU peak mem in MB", "backend", axes[0, 0])
    draw_error_bar(df2d, "degree of freedom", "GPU peak mem in MB", "backend", axes[0, 1])
    draw_error_bar(df3d, "degree of freedom", "CPU peak mem in MB", "backend", axes[1, 0])
    draw_error_bar(df3d, "degree of freedom", "GPU peak mem in MB", "backend", axes[1, 1])
    axes[0, 0].set_title("(a)")
    axes[0, 1].set_title("(b)")
    axes[1, 0].set_title("(c)")
    axes[1, 1].set_title("(d)")

    fig.savefig("compare_linear_poisson_mem.png")
    fig.savefig("compare_linear_poisson_mem.pdf")
    fig.savefig("compare_linear_poisson_mem.eps")