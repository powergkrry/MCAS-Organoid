import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import seaborn as sns

#
organoid_stats_24_wells = {
    "D35": {"organoids_area_in_pixel": []},
    "D56": {"organoids_area_in_pixel": []},
    "D70": {"organoids_area_in_pixel": []},
    "D84": {"organoids_area_in_pixel": []},
}

#
root_path = "./"
out_path = os.path.join(
    root_path, "different_sizes_24_well_all_final_tiled_focal_stack_0_mask.nc"
)
mask_dataset = xr.load_dataset(out_path, engine="netcdf4", mask_and_scale=False)
imaginary_eagle_structure = mask_dataset.images[0, 0].data

# all 24 wellplates
sub_row_arrays_keys = ["D84", "D84", "D84", "D70", "D56", "D35"]
imaginary_eagle_structure_sub_row_arrays = np.split(
    imaginary_eagle_structure, [24800, 64000, 103200, 142400, 181600], axis=0
)

for i, (sub_row_arrays_key, imaginary_eagle_structure_sub_row_array) in tqdm(
    enumerate(zip(sub_row_arrays_keys, imaginary_eagle_structure_sub_row_arrays))
):
    # remove cropped wells
    if i == 0:
        continue

    imaginary_eagle_structure_sub_row_col_arrays = np.split(
        imaginary_eagle_structure_sub_row_array, [33600, 73600, 113600], axis=1
    )

    for (
        imaginary_eagle_structure_sub_row_col_array
    ) in imaginary_eagle_structure_sub_row_col_arrays:
        kernel = np.ones((5, 5), np.uint8)
        imaginary_eagle_structure_sub_row_col_array_dilate = cv2.dilate(
            imaginary_eagle_structure_sub_row_col_array, kernel, iterations=1
        )

        cnts = cv2.findContours(
            imaginary_eagle_structure_sub_row_col_array_dilate,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # get contour mask
        for cnt in cnts:
            mask = np.zeros_like(imaginary_eagle_structure_sub_row_col_array_dilate)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            # pass cropped organoids at the top
            if i == 0 and np.any(mask[1000] != 0):
                continue

            contour, hier = cv2.findContours(
                mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contour:
                cv2.drawContours(mask, [cnt], 0, 255, -1)

            organoid_area_in_pixel = np.sum(mask) / 255
            organoid_stats_24_wells[sub_row_arrays_key][
                "organoids_area_in_pixel"
            ].append(organoid_area_in_pixel)
cluster_stats_6_wells_save_path = os.path.join(
    root_path, "organoid_stats_24_wells.json"
)
with open(
    cluster_stats_6_wells_save_path,
    "w",
    encoding="utf-8",
) as f:
    base_info = json.dump(organoid_stats_24_wells, f, ensure_ascii=False, indent=4)


# for plot
cluster_stats_6_wells_save_path = os.path.join(
    root_path, "organoid_stats_24_wells.json"
)
with open(cluster_stats_6_wells_save_path, "r", encoding="utf-8") as f:
    cluster_stats_6_wells = json.load(f)

# remove really small masks
threshold = 100000
sub_row_arrays_keys = ["D35", "D56", "D70", "D84"]
for sub_row_arrays_key in sub_row_arrays_keys:
    cells = np.array(
        cluster_stats_6_wells[sub_row_arrays_key]["organoids_area_in_pixel"]
    )
    filtered_cells = cells[cells >= threshold]
    cluster_stats_6_wells[sub_row_arrays_key][
        "organoids_area_in_pixel"
    ] = filtered_cells

#
group1 = cluster_stats_6_wells["D35"]["organoids_area_in_pixel"] / (4 * 1000 * 1000)
group2 = cluster_stats_6_wells["D56"]["organoids_area_in_pixel"] / (4 * 1000 * 1000)
group3 = cluster_stats_6_wells["D70"]["organoids_area_in_pixel"] / (4 * 1000 * 1000)
group4 = cluster_stats_6_wells["D84"]["organoids_area_in_pixel"] / (4 * 1000 * 1000)

data = np.concatenate([group1, group2, group3, group4])

labels = (
    ["D35"] * len(group1)
    + ["D56"] * len(group2)
    + ["D70"] * len(group3)
    + ["D84"] * len(group4)
)

# Create DataFrame for seaborn
df = pd.DataFrame({"Group": labels, "Size": data})

# Set custom colors for each group
colorblind_palette = sns.color_palette("colorblind", 4)

# Violin plot
sns.violinplot(
    x="Group",
    y="Size",
    data=df,
    palette=colorblind_palette,
    inner=None,
    scale="count",
    cut=0,
    common_norm=False,
)

# # Add data points
# sns.swarmplot(x="Group", y="Size", data=df, color="black", size=3)

# Calculate means
group_order = ["D35", "D56", "D70", "D84"]
means = [df[df["Group"] == g]["Size"].mean() for g in group_order]

# Add mean lines
for i, mean in enumerate(means):
    plt.plot(
        [i - 0.4, i + 0.4],
        [mean, mean],
        color="k",
        linewidth=2,
        label="Mean" if i == 0 else "",
    )

plt.title("Organoids size")
plt.show()
