import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


#
root_path = "./"
out_path = os.path.join(root_path, "mcam_mask_dataset.nc")
mask_dataset = xr.load_dataset(out_path, engine="netcdf4", mask_and_scale=False)
out_path = os.path.join(root_path, "mcam_mask_dataset_FL.nc")
mask_dataset_FL = xr.load_dataset(out_path, engine="netcdf4", mask_and_scale=False)

#
num_of_cells = np.full((2, 6 * 2, 8), 0)

for y, x in np.ndindex(12, 8):
    # count # of cells
    num_of_cells[0, y, x] = len(pd.unique(np.ravel(mask_dataset_FL.images[y, x])))
    num_of_cells[1, y, x] = len(pd.unique(np.ravel(mask_dataset.images[y, x])))
np.save(os.path.join(root_path, "num_of_cells.npy"), num_of_cells)

#
out_path = os.path.join(root_path, "mcam_max_image_dataset.nc")
image_dataset = xr.load_dataset(out_path, engine="netcdf4", mask_and_scale=False)

#
BF_FL_cell_confluency = np.full((3, 6 * 2, 8), 0)

for y, x in np.ndindex(12, 8):
    # count # of cells
    BF_FL_cell_confluency[0, y, x] = np.sum(mask_dataset.images[y, x] != 0)
    BF_FL_cell_confluency[1, y, x] = np.sum(mask_dataset_FL.images[y, x] != 0)
    BF_FL_cell_confluency[2, y, x] = np.sum(
        image_dataset.images[y, x] > 35
    )  # seems 35 segments well boundary well
np.save(os.path.join(root_path, "BF_FL_cell_confluency.npy"), BF_FL_cell_confluency)

#
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

mpl.rcParams["font.family"] = "DejaVu Sans"  # bundled with matplotlib
# optional: ensure sans-serif resolves to DejaVu
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]

#
root_path = "./"
num_of_cells = np.load(os.path.join(root_path, "num_of_cells.npy"))

percentage_cells = num_of_cells[0] / num_of_cells[1]
percentage_cells = np.rot90(percentage_cells * 100, 1)

transfections = percentage_cells.ravel()

# Normalize confluency values for colormap (0 to max)
norm = mcolors.Normalize(vmin=np.min(transfections), vmax=np.max(transfections))
# cmap = cm.get_cmap('plasma')  # Colorblind-friendly colormap
# Define the colors for the colormap
colors = ["black", "green"]
# Create the custom colormap
cmap = LinearSegmentedColormap.from_list("BlackGreen", colors)


# Helper to decide text color based on background
def get_contrasting_text_color(rgb):
    # Convert RGB to perceived brightness (0–1)
    r, g, b = rgb[:3]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if brightness > 0.6 else "white"


# Plot setup
fig, ax = plt.subplots(figsize=(14, 8.2))  # Larger figure to fit 96 wells
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.set_xlim(0, 12)  # 12 columns
ax.set_ylim(0, 8)  # 8 rows
ax.set_xticks(np.arange(0.5, 12, 1))
ax.set_yticks(np.arange(0.5, 8, 1))

# Draw wells (8 rows x 12 columns = 96 wells)
for i in range(96):
    col = i % 12
    row = 7 - (
        i // 12
    )  # Reverse row order to match visual representation (top to bottom)
    color = cmap(norm(transfections[i]))
    text_color = get_contrasting_text_color(color)
    circle = plt.Circle((col + 0.5, row + 0.5), 0.4, color=color, ec="black")
    ax.add_patch(circle)
    # ax.text(col + 0.5, row + 0.5, f"{transfections[i]:.1f}%",
    #         ha='center', va='center', fontsize=13, color=text_color, weight='bold')

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
cbar.ax.tick_params(labelsize=40)
cbar.set_label(label="Transduction efficiency (%)", size=40)

# ax.set_title("96-Well Plate Transfection", fontsize=60)
ax.set_xticklabels(np.arange(1, 13), fontsize=40)  # Label columns 1 to 12
ax.set_yticklabels(
    ["H", "G", "F", "E", "D", "C", "B", "A"], fontsize=40
)  # Label rows 1 to 8
plt.tight_layout()
plt.show()


#
root_path = "./"
BF_FL_cell_confluency = np.load(os.path.join(root_path, "BF_FL_cell_confluency.npy"))

BF_confluency = BF_FL_cell_confluency[0] / BF_FL_cell_confluency[2] * 100
FL_confluency = BF_FL_cell_confluency[1] / BF_FL_cell_confluency[2] * 100

BF_confluency = BF_confluency.ravel()
FL_confluency = FL_confluency.ravel()

# BF
# Normalize confluency values for colormap (0 to max)
norm = mcolors.Normalize(vmin=np.min(BF_confluency), vmax=np.max(BF_confluency))
# cmap = cm.get_cmap('plasma')  # Colorblind-friendly colormap
# Define the colors for the colormap
colors = ["black", "green"]
# Create the custom colormap
cmap = LinearSegmentedColormap.from_list("BlackGreen", colors)


# Helper to decide text color based on background
def get_contrasting_text_color(rgb):
    # Convert RGB to perceived brightness (0–1)
    r, g, b = rgb[:3]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if brightness > 0.6 else "white"


# Plot setup
fig, ax = plt.subplots(figsize=(14, 8.2))  # Larger figure to fit 96 wells
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.set_xlim(0, 12)  # 12 columns
ax.set_ylim(0, 8)  # 8 rows
ax.set_xticks(np.arange(0.5, 12, 1))
ax.set_yticks(np.arange(0.5, 8, 1))

# Draw wells (8 rows x 12 columns = 96 wells)
for i in range(96):
    col = i % 12
    row = 7 - (
        i // 12
    )  # Reverse row order to match visual representation (top to bottom)
    color = cmap(norm(BF_confluency[i]))
    text_color = get_contrasting_text_color(color)
    circle = plt.Circle((col + 0.5, row + 0.5), 0.4, color=color, ec="black")
    ax.add_patch(circle)
    # ax.text(col + 0.5, row + 0.5, f"{BF_confluency[i]:.1f}%",
    #         ha='center', va='center', fontsize=13, color=text_color, weight='bold')

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
cbar.ax.tick_params(labelsize=40)
cbar.set_label(label="Confluency (%)", size=40)

# ax.set_title("96-Well Plate BF Confluency", fontsize=30)
ax.set_xticklabels(np.arange(1, 13), fontsize=40)  # Label columns 1 to 12
ax.set_yticklabels(
    ["H", "G", "F", "E", "D", "C", "B", "A"], fontsize=40
)  # Label rows 1 to 8
plt.tight_layout()
plt.show()
