import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr


#
root_path = "./"
dates_to_process = [
    "20241008",
    "20241009",
    "20241010",
    "20241011",
    "20241012",
    "20241013",
]
media_change_dates = []
num_dose = 2

# gather average number of pixel
mask_num_pixel_LR_array_temp = np.zeros((2, len(dates_to_process), 12, 8))
for plate_number in range(2):
    out_path = os.path.join(root_path, f"plate{plate_number+1}_mcam_mask_dataset.nc")
    mask_dataset = xr.load_dataset(out_path, engine="netcdf4", mask_and_scale=False)

    for date_number, date_to_process in enumerate(dates_to_process):
        for y, x in np.ndindex(12, 8):
            mask_num_pixel_LR_array_temp[plate_number, date_number, y, x] = (
                mask_dataset["mask_num_pixel"][date_number, y, x]
            )

# reshape average number of pixel
mask_num_pixel_LR_array = np.zeros((len(dates_to_process), 12 * 2, 8))

mask_num_pixel_LR_array[:, 0:6, :] = mask_num_pixel_LR_array_temp[0, :, 0:6, :]
mask_num_pixel_LR_array[:, 6:12, :] = mask_num_pixel_LR_array_temp[1, :, 0:6, :]
mask_num_pixel_LR_array[:, 12:18, :] = mask_num_pixel_LR_array_temp[0, :, 6:12, :]
mask_num_pixel_LR_array[:, 18:24, :] = mask_num_pixel_LR_array_temp[1, :, 6:12, :]

# calculate average number of pixel
average_num_pixel_array = np.zeros((len(dates_to_process), num_dose + 1))
std_num_pixel_array = np.zeros((len(dates_to_process), num_dose + 1))

# for 20% and 40%
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

# for 100%
for date_number, date_to_process in enumerate(dates_to_process):
    average_num_pixel_array[date_number, dose + 1] = np.nanmean(
        mask_num_pixel_LR_array[date_number, 12:, :]
    )
    std_num_pixel_array[date_number, dose + 1] = np.nanstd(
        mask_num_pixel_LR_array[date_number, 12:, :]
    )

# convert # of pixel to size (mm2)
average_num_pixel_array /= 4 * 1000 * 1000
std_num_pixel_array /= 4 * 1000 * 1000

# calculate standard error
num_included_organoids = [48, 48, 96]
for dose in range(num_dose + 1):
    std_num_pixel_array[:, dose] /= np.sqrt(num_included_organoids[dose])

# plot
x = np.array(range(len(dates_to_process)))
day_for_plot = [f"Day {i}" for i in np.array(x) + 1]
color = ["#56B4E9", "#E69F00", "#009E73"]
VPA_dose = ["20%", "40%", "100%"]
num_included_organoids = ["48", "48", "96"]

plt.figure(figsize=(9, 9))
# plt.title('Drug response curve')
plt.xlabel("Differentiation date", fontsize=16)
plt.ylabel("Average size ($mm^2$)", fontsize=16)
for dose in range(num_dose + 1):
    plt.errorbar(
        x,
        average_num_pixel_array[:, dose],
        yerr=std_num_pixel_array[:, dose],
        label=f"{VPA_dose[dose]}, {num_included_organoids[dose]} organoids",
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

# plot
x = list(range(len(dates_to_process)))
day_for_plot = [f"Day {i}" for i in np.array(x) + 1]
color = ["#56B4E9", "#E69F00", "#009E73"]
VPA_dose = ["20%", "40%", "100%"]
num_included_organoids = ["48", "48", "96"]

# 20% and 100 %
out_path = os.path.join(root_path, "plate1_mcam_max_image_FL_dataset.nc")
mask_dataset = xr.load_dataset(out_path, engine="netcdf4", mask_and_scale=False)

exposure_time_max = np.max(mask_dataset.exposure_time.data)
exposure_time_20 = mask_dataset.exposure_time[:, 0:6, :].data
sum_of_signal_20 = mask_dataset.sum_of_signal[:, 0:6, :].data
sum_of_all_20 = mask_dataset.sum_of_all[:, 0:6, :].data
exposure_time_100 = mask_dataset.exposure_time[:, 6:, :].data
sum_of_signal_100 = mask_dataset.sum_of_signal[:, 6:, :].data
sum_of_all_100 = mask_dataset.sum_of_all[:, 6:, :].data

# 40% and 100 %
out_path = os.path.join(root_path, "plate2_mcam_max_image_FL_dataset.nc")
mask_dataset = xr.load_dataset(out_path, engine="netcdf4", mask_and_scale=False)

exposure_time_40 = mask_dataset.exposure_time[:, 0:6, :].data
sum_of_signal_40 = mask_dataset.sum_of_signal[:, 0:6, :].data
sum_of_all_40 = mask_dataset.sum_of_all[:, 0:6, :].data
exposure_time_100 = np.stack(
    (exposure_time_100, mask_dataset.exposure_time[:, 6:, :].data), axis=0
)
sum_of_signal_100 = np.stack(
    (sum_of_signal_100, mask_dataset.sum_of_signal[:, 6:, :].data), axis=0
)
sum_of_all_100 = np.stack(
    (sum_of_all_100, mask_dataset.sum_of_all[:, 6:, :].data), axis=0
)

plt.figure(figsize=(9, 9))
plt.title("STICR", fontsize=16)
plt.xlabel("Differentiation date", fontsize=16)
plt.ylabel("Signal", fontsize=16)
plt.errorbar(
    x,
    np.mean(
        sum_of_signal_20 * (exposure_time_max / exposure_time_20) / sum_of_all_20,
        axis=(1, 2),
    ),
    yerr=np.std(
        sum_of_signal_20 * (exposure_time_max / exposure_time_20) / sum_of_all_20,
        axis=(1, 2),
    )
    / np.sqrt(48),
    label=f"{VPA_dose[0]}, {num_included_organoids[0]} organoids",
    fmt="o-",
    capsize=5,
    capthick=2,
    color=color[0],
    marker="s",
    linewidth=2,
)
plt.errorbar(
    x,
    np.mean(
        sum_of_signal_40 * (exposure_time_max / exposure_time_40) / sum_of_all_40,
        axis=(1, 2),
    ),
    yerr=np.std(
        sum_of_signal_40 * (exposure_time_max / exposure_time_40) / sum_of_all_40,
        axis=(1, 2),
    )
    / np.sqrt(48),
    label=f"{VPA_dose[1]}, {num_included_organoids[1]} organoids",
    fmt="o-",
    capsize=5,
    capthick=2,
    color=color[1],
    marker="s",
    linewidth=2,
)
plt.errorbar(
    x,
    np.mean(
        sum_of_signal_100 * (exposure_time_max / exposure_time_100) / sum_of_all_100,
        axis=(0, 2, 3),
    ),
    yerr=np.std(
        sum_of_signal_100 * (exposure_time_max / exposure_time_100) / sum_of_all_100,
        axis=(0, 2, 3),
    )
    / np.sqrt(96),
    label=f"{VPA_dose[2]}, {num_included_organoids[2]} organoids",
    fmt="o-",
    capsize=5,
    capthick=2,
    color=color[2],
    marker="s",
    linewidth=2,
)
plt.yticks(fontsize=14)
plt.xticks(x, day_for_plot, rotation=45, fontsize=14)
plt.legend(loc="upper left", fontsize=16)
plt.tight_layout()
