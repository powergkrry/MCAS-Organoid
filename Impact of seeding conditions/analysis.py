import json
import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr


root_path = "./"
dates_to_process = [
    "20240207",
    "20240208",
    "20240209",
    "20240210",
    "20240211",
    "20240212",
    "20240213",
    "20240214",
    "20240215",
    "20240216",
    "20240217",
    "20240218",
    "20240219",
    "20240220",
    "20240221",
    "20240222",
    "20240223",
]
media_change_dates = ["20240209", "20240212", "20240215", "20240218", "20240221"]
num_dose = 3

# determine which well will be included in the final analysis
include_to_analysis_array = np.full((12, 8), True)
for date_number, date_to_process in enumerate(dates_to_process):
    for y, x in np.ndindex(12, 8):
        if y < 6:
            plate_L_or_R = "L"
        else:
            plate_L_or_R = "R"

        # load base information
        with open(
            os.path.join(
                root_path,
                "base_info",
                date_to_process,
                f"base_info_{plate_L_or_R}.json",
            ),
            "r",
            encoding="utf-8",
        ) as f:
            base_info = json.load(f)

        y_adjusted = y - 6 if plate_L_or_R == "R" else y
        include_to_analysis_array[y, x] = np.logical_and(
            include_to_analysis_array[y, x],
            base_info[f"({x},{y_adjusted})"]["include_to_analysis"],
        )  # switched

# gather average number of pixel
mask_num_pixel_LR_array = np.zeros((len(dates_to_process), 12, 8))
eccentricity_LR_array = np.zeros((len(dates_to_process), 12, 8))

out_path = os.path.join(root_path, "mcam_mask_dataset.nc")
mask_dataset = xr.load_dataset(out_path, engine="netcdf4", mask_and_scale=False)

for date_number, date_to_process in enumerate(dates_to_process):
    for y, x in np.ndindex(12, 8):
        mask_num_pixel_LR_array[date_number, y, x] = mask_dataset["mask_num_pixel"][
            date_number, y, x
        ]
        eccentricity_LR_array[date_number, y, x] = mask_dataset["eccentricity"][
            date_number, y, x
        ]

include_to_analysis_array_expanded = np.expand_dims(include_to_analysis_array, axis=0)
include_to_analysis_array_expanded = np.repeat(
    include_to_analysis_array_expanded, len(dates_to_process), axis=0
)
mask_num_pixel_LR_array[~include_to_analysis_array_expanded] = None
eccentricity_LR_array[~include_to_analysis_array_expanded] = None

# calculate average number of pixel
average_num_pixel_array = np.zeros((len(dates_to_process), num_dose))
std_num_pixel_array = np.zeros((len(dates_to_process), num_dose))
average_eccentricity_array = np.zeros((len(dates_to_process), num_dose))
std_eccentricity_array = np.zeros((len(dates_to_process), num_dose))
for dose in range(num_dose):
    for date_number, date_to_process in enumerate(dates_to_process):
        average_num_pixel_array[date_number, dose] = np.nanmean(
            mask_num_pixel_LR_array[
                date_number, dose * (12 // num_dose) : (dose + 1) * (12 // num_dose), :
            ]
        )
        std_num_pixel_array[date_number, dose] = np.nanstd(
            mask_num_pixel_LR_array[
                date_number, dose * (12 // num_dose) : (dose + 1) * (12 // num_dose), :
            ]
        )
        average_eccentricity_array[date_number, dose] = np.nanmean(
            eccentricity_LR_array[
                date_number, dose * (12 // num_dose) : (dose + 1) * (12 // num_dose), :
            ]
        )
        std_eccentricity_array[date_number, dose] = np.nanstd(
            eccentricity_LR_array[
                date_number, dose * (12 // num_dose) : (dose + 1) * (12 // num_dose), :
            ]
        )

# convert # of pixel to size (mm2)
average_num_pixel_array /= 4 * 1000 * 1000
std_num_pixel_array /= 4 * 1000 * 1000

# calculate standard error
for dose in range(num_dose):
    num_included_organoids = np.sum(
        include_to_analysis_array[
            dose * (12 // num_dose) : (dose + 1) * (12 // num_dose)
        ]
    )
    std_num_pixel_array[:, dose] /= np.sqrt(num_included_organoids)
    std_eccentricity_array[:, dose] /= np.sqrt(num_included_organoids)

# plot
x = list(range(len(dates_to_process)))
day_for_plot = [f"Day {i}" for i in np.array(x) + 1]
color = ["#56B4E9", "#E69F00", "#009E73"]

plt.figure(figsize=(9, 9))
# plt.title('Effect of different seeding medias')
plt.xlabel("Days post differentiation", fontsize=16)
plt.ylabel("Average size ($mm^2$)", fontsize=16)
for dose in range(num_dose):
    num_included_organoids = np.sum(
        include_to_analysis_array[
            dose * (12 // num_dose) : (dose + 1) * (12 // num_dose)
        ]
    )
    plt.errorbar(
        x,
        average_num_pixel_array[:, dose],
        yerr=std_num_pixel_array[:, dose],
        label=f"Condition {dose+1}, {num_included_organoids} organoids",
        fmt="o-",
        capsize=5,
        capthick=2,
        color=color[dose],
        marker="s",
        linewidth=2,
    )

plt.yticks(fontsize=14)
plt.xticks(x, day_for_plot, rotation=45, fontsize=14)
for media_change_date in media_change_dates:
    plt.axvspan(
        dates_to_process.index(media_change_date) - 0.8,
        dates_to_process.index(media_change_date) - 0.2,
        alpha=0.2,
        color="blue",
        label="Media change",
    )
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=16)
plt.tight_layout()

# plot - average_eccentricity_array
x = list(range(len(dates_to_process)))
day_for_plot = [f"Day {i}" for i in np.array(x) + 1]
color = ["#56B4E9", "#E69F00", "#009E73"]

plt.figure(figsize=(9, 9))
# plt.title('Eccentricity of organoids')
plt.xlabel("Days post differentiation", fontsize=16)
plt.ylabel("Average eccentricity", fontsize=16)
for dose in range(num_dose):
    num_included_organoids = np.sum(
        include_to_analysis_array[
            dose * (12 // num_dose) : (dose + 1) * (12 // num_dose)
        ]
    )
    plt.errorbar(
        x,
        average_eccentricity_array[:, dose],
        yerr=std_eccentricity_array[:, dose],
        label=f"Condition {dose+1}, {num_included_organoids} organoids",
        fmt="o-",
        capsize=5,
        capthick=2,
        color=color[dose],
        marker="s",
        linewidth=2,
    )

plt.yticks(fontsize=14)
plt.xticks(x, day_for_plot, rotation=45, fontsize=14)
for media_change_date in media_change_dates:
    plt.axvspan(
        dates_to_process.index(media_change_date) - 0.8,
        dates_to_process.index(media_change_date) - 0.2,
        alpha=0.2,
        color="blue",
        label="Media change",
    )
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=16)
plt.tight_layout()
