import json
import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr


root_path = "./"
dates_to_process = [
    "20240517",
    "20240518",
    "20240519",
    "20240520",
    "20240521",
    "20240522",
    "20240523",
    "20240524",
    "20240525",
    "20240526",
    "20240527",
    "20240528",
    "20240529",
    "20240530",
    "20240531",
]
drug_added_date = "20240517"
media_change_dates = ["20240518", "20240520", "20240523", "20240526", "20240529"]
num_dose = 6

# determine which well will be included in the final analysis
include_to_analysis_array = np.full((2, 12, 8), True)
for plate_number in range(2):
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
                    f"plate{plate_number+1}_base_info_stitched_{plate_L_or_R}.json",
                ),
                "r",
                encoding="utf-8",
            ) as f:
                base_info = json.load(f)

            y_adjusted = y - 6 if plate_L_or_R == "R" else y
            include_to_analysis_array[plate_number, y, x] = np.logical_and(
                include_to_analysis_array[plate_number, y, x],
                base_info[f"({x},{y_adjusted})"]["include_to_analysis"],
            )  # switched

# gather average number of pixel
mask_num_pixel_LR_array = np.zeros((2, len(dates_to_process), 12, 8))
for plate_number in range(2):
    out_path = os.path.join(
        root_path, f"plate{plate_number+1}_stitched_mcam_mask_dataset.nc"
    )
    mask_dataset = xr.load_dataset(out_path, engine="netcdf4", mask_and_scale=False)

    for date_number, date_to_process in enumerate(dates_to_process):
        for y, x in np.ndindex(12, 8):
            mask_num_pixel_LR_array[plate_number, date_number, y, x] = mask_dataset[
                "mask_num_pixel"
            ][date_number, y, x]

include_to_analysis_array_expanded = np.expand_dims(include_to_analysis_array, axis=1)
include_to_analysis_array_expanded = np.repeat(
    include_to_analysis_array_expanded, len(dates_to_process), axis=1
)
mask_num_pixel_LR_array[~include_to_analysis_array_expanded] = None

# control group is located at the edges. shift for easier coding
# this will locate the control at the begining of the array
mask_num_pixel_LR_array = np.roll(mask_num_pixel_LR_array, 1, axis=2)

# calculate average number of pixel
average_num_pixel_array = np.zeros((2, len(dates_to_process), num_dose))
std_num_pixel_array = np.zeros((2, len(dates_to_process), num_dose))
for plate_number in range(2):
    for dose in range(num_dose):
        for date_number, date_to_process in enumerate(dates_to_process):
            average_num_pixel_array[plate_number, date_number, dose] = np.nanmean(
                mask_num_pixel_LR_array[
                    plate_number,
                    date_number,
                    dose * (12 // num_dose) : (dose + 1) * (12 // num_dose),
                    :,
                ]
            )
            std_num_pixel_array[plate_number, date_number, dose] = np.nanstd(
                mask_num_pixel_LR_array[
                    plate_number,
                    date_number,
                    dose * (12 // num_dose) : (dose + 1) * (12 // num_dose),
                    :,
                ]
            )

# convert # of pixel to size (mm2)
average_num_pixel_array /= 4 * 1000 * 1000
std_num_pixel_array /= 4 * 1000 * 1000

# calculate standard error
for plate_number in range(2):
    for dose in range(num_dose):
        num_included_organoids = np.sum(
            include_to_analysis_array[
                plate_number, dose * (12 // num_dose) : (dose + 1) * (12 // num_dose)
            ]
        )
        std_num_pixel_array[plate_number, :, dose] /= np.sqrt(num_included_organoids)

# plot
x = np.array(range(len(dates_to_process)))
day_for_plot = [f"Day {i}" for i in np.array(x) + 16 + 1]
color = ["#56B4E9", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]
VPA_dose = [
    [
        "Water Control",
        "Mitomycin 0.5ug/ml for 24 hrs",
        "Mitomycin 1ug/ml for 24 hrs",
        "Mitomycin 1ug/ml for 1 hr",
        "Mitomycin 3ug/ml for 1 hr",
        "Lithium 3mM",
    ],
    [
        "Vehicle Control",
        "CHIR 0.625uM",
        "CHIR 1.25uM",
        "CHIR 2.5uM",
        "VPA 0.6mM",
        "VPA 3mM",
    ],
]

for plate_number in range(2):
    plt.figure(figsize=(9, 9))
    plt.title("Drug response curve", fontsize=16)
    plt.xlabel("Days post differentiation", fontsize=16)
    plt.ylabel("Average size ($mm^2$)", fontsize=16)
    for dose in range(num_dose):
        # skip Lithium
        if plate_number == 0 and dose == 5:
            continue

        # num_included_organoids = np.sum(include_to_analysis_array[plate_number,dose*(12//num_dose):(dose+1)*(12//num_dose),:])
        plt.errorbar(
            x,
            average_num_pixel_array[plate_number, :, dose],
            yerr=std_num_pixel_array[plate_number, :, dose],
            label=f"{VPA_dose[plate_number][dose]}",
            fmt="o-",
            capsize=5,
            capthick=2,
            color=color[dose],
            marker="s",
            linewidth=2,
        )
    plt.yticks(fontsize=14)
    plt.xticks(x, day_for_plot, rotation=45, fontsize=14)
    plt.axvspan(
        dates_to_process.index(drug_added_date) - 0.2,
        dates_to_process.index(drug_added_date) + 0.2,
        alpha=0.3,
        color="red",
        label="Drug added",
    )
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
    if plate_number == 0:
        plt.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            fontsize=16,
            bbox_to_anchor=(0.42, 0.6),
        )
    else:
        plt.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=16)
    plt.tight_layout()
