import json
import numpy as np
import os
import cv2
import xarray as xr
import glob
from owl.mcam_data import new_dataset, new_rgb_dataset
from owl import mcam_data
from tqdm import tqdm
from owl.instruments.falcon import alloc_buffer_for_acquisition
import tifffile as tiff
from skimage.measure import perimeter
from skimage.measure import EllipseModel
import copy

def new_dataset_with_frame_number(z_num_steps, image_shape=3072):
    y_dim = 8
    x_dim = 6
    single_image_dataset = new_dataset(N_cameras=(y_dim,x_dim),
                                       image_shape=(image_shape,image_shape))

    stacked_vars = ['images']
    full_dataset = single_image_dataset.assign_coords({
        'frame_number': np.arange(z_num_steps, dtype='int'),
    })
    for name in stacked_vars:
        var = single_image_dataset[name]
        dims = ('frame_number',) + var.dims
        shape = (z_num_steps,) + var.shape
        if name == 'images':
            data = alloc_buffer_for_acquisition(
                shape,
                dtype=var.dtype
            )
        else:
            data = np.empty(shape, dtype=var.dtype)
        full_dataset[name] = xr.DataArray(
            data=data,
            dims=dims
        )
    
    return full_dataset

def new_rgb_dataset_with_frame_number(z_num_steps, image_shape=3072):
    y_dim = 6*2 # 8 # switched!
    x_dim = 8 # 6*2 # switched!
    
    single_image_dataset = new_rgb_dataset(N_cameras=(y_dim,x_dim),
                                       image_shape=(image_shape,image_shape,3))

    stacked_vars = ['images']
    full_dataset = single_image_dataset.assign_coords({
        'frame_number': np.arange(z_num_steps, dtype='int'),
    })
    for name in stacked_vars:
        var = single_image_dataset[name]
        dims = ('frame_number',) + var.dims
        shape = (z_num_steps,) + var.shape
        if name == 'images':
            data = alloc_buffer_for_acquisition(
                shape,
                dtype=var.dtype
            )
        else:
            data = np.empty(shape, dtype=var.dtype)
        full_dataset[name] = xr.DataArray(
            data=data,
            dims=dims
        )
    
    return full_dataset

def calculate_circularity(mask):
    area = np.sum(mask)
    if area == 0:
        return np.nan

    perimeter_value = perimeter(mask)

    circularity = (4 * np.pi * area) / perimeter_value**2

    return circularity


#%%
root_path = '/media/kanghyun/Sunday/Temporary/kanghyun/Eagle/83_UNC_STICR_FL'
dates_to_process = ['20241008', '20241009', '20241010', '20241011', '20241012', '20241013']
experiment = 'KH_D'
if not os.path.exists(os.path.join(root_path, experiment, 'analysis')):
    os.makedirs(os.path.join(root_path, experiment, 'analysis'))


#%%
checker = np.full((2,2*len(dates_to_process)*8*6), False)


# %% get max projection images - BF
for plate_number in range(2):
    i = 0
    
    if not any(checker[plate_number]):
        max_image_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process), image_shape=3072)
        
    for plate_L_or_R in ['L', 'R']:
        
        for date_number, date_to_process in tqdm(enumerate(dates_to_process)):
            data = mcam_data.load(glob.glob(os.path.join(root_path, experiment,  date_to_process,
                                                          f'Day{date_number+1}_Plate{plate_number+1}_{plate_L_or_R}',
                                                          'center_stack*', 'mcam_dataset.nc'))[0], engine='ramona')
            
            # load shift information
            shift_file_name = 'shift_info_20241016.json'
            with open(os.path.join(root_path, shift_file_name), 'r', encoding='utf-8') as f:
                shift_info = json.load(f)
            
            for y,x in np.ndindex(8,6):
                if checker[plate_number,i]:
                    i += 1
                    continue
                print(y,x)
                
                x_to_save = x+6 if plate_L_or_R == 'R' else x
                
                row_shift_per_stack = shift_info[f'({y},{x})']['row_shift_per_px']
                col_shift_per_stack = shift_info[f'({y},{x})']['col_shift_per_px']
                
                data_subset_gray_array = np.zeros((data.images.shape[0], 3072, 3072), dtype=np.uint8)
                
                for z_stack in range(data.images.shape[0]):
                    data_subset_gray_array[z_stack] = cv2.cvtColor(data.images[z_stack,y,x].values,
                                                                   cv2.COLOR_BayerGR2GRAY)
                    data_subset_gray_array[z_stack] = np.roll(data_subset_gray_array[z_stack],
                                                     (int(row_shift_per_stack*z_stack),
                                                      int(col_shift_per_stack*z_stack)),
                                                     axis=(0,1))
                data_subset_gray_array_max_projection = np.max(data_subset_gray_array, axis=0)
                data_subset_gray_array_max_projection = np.repeat(data_subset_gray_array_max_projection[..., np.newaxis], 3, axis=2)
                max_image_dataset['images'][date_number,x_to_save,y] = data_subset_gray_array_max_projection # switched!
                
                checker[plate_number,i] = True
                i += 1
    mcam_data.save(max_image_dataset, os.path.join(root_path, experiment, 'analysis', f'plate{plate_number+1}_mcam_max_image_dataset.nc'),
                    include_timestamp=False, engine='ramona')


#%% generate .nc for manually annotated data
for plate_number in range(2):
    max_image_dataset = mcam_data.load(os.path.join(root_path, experiment, 'analysis',
                                                    f'plate{plate_number+1}_mcam_max_image_dataset.nc'), engine='ramona')
    contour_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process), image_shape=3072)
    mask_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process), image_shape=3072)
    
    mask_num_pixel = np.full((len(dates_to_process),6*2,8), 0) # switched!
    circularity = np.full((len(dates_to_process),6*2,8), 0.) # switched!
    eccentricity = np.full((len(dates_to_process),6*2,8), 0.) # switched!
    smoothness = np.full((len(dates_to_process),6*2,8), 0.) # switched!
    mask_num_pixel_dataarray = xr.DataArray(mask_num_pixel,
                                            dims=['frame_number', 'image_y', 'image_x'],
                                            coords=[range(len(dates_to_process)), range(6*2), range(8)]) # switched!
    circularity_dataarray = xr.DataArray(circularity,
                                          dims=['frame_number', 'image_y', 'image_x'],
                                          coords=[range(len(dates_to_process)), range(6*2), range(8)]) # switched!
    eccentricity_dataarray = xr.DataArray(eccentricity,
                                          dims=['frame_number', 'image_y', 'image_x'],
                                          coords=[range(len(dates_to_process)), range(6*2), range(8)]) # switched!
    smoothness_dataarray = xr.DataArray(smoothness,
                                          dims=['frame_number', 'image_y', 'image_x'],
                                          coords=[range(len(dates_to_process)), range(6*2), range(8)]) # switched!
    mask_dataset['mask_num_pixel'] = mask_num_pixel_dataarray
    mask_dataset['circularity'] = circularity_dataarray
    mask_dataset['eccentricity'] = eccentricity_dataarray
    mask_dataset['smoothness'] = smoothness_dataarray
    
    for date_number, date_to_process in tqdm(enumerate(dates_to_process)):
        for y,x in np.ndindex(12,8):
            data_subset_gray_array_max_projection = np.array(max_image_dataset['images'][date_number,y,x])
            mask = tiff.imread(os.path.join(root_path, experiment, 'analysis', 'annotation', 'done',
                                            f'plate{plate_number+1}_frame{date_number:02d}_y{y:02d}_x{x:02d}_mask.tif')).astype(np.uint8)
            
            # draw contours
            image_and_contour = copy.deepcopy(data_subset_gray_array_max_projection)
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            # find the max length contour
            c = max(cnts, key = len)
            cv2.drawContours(image_and_contour, [c], -1, (36, 255, 12), thickness=15)
            
            # get final mask from the max contour
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # get ellipse boundary
            ellipse_matrix = np.sum(image_and_contour==[36,255,12], axis=2) == 3
            # get ellipse boundary coordinate
            res = np.array(list(zip(*np.nonzero(ellipse_matrix))))
            # Fit the ellipse
            ellipse = EllipseModel()
            ellipse.estimate(res)
            # Get the ellipse parameters
            xc, yc, a, b, theta = ellipse.params
            eccentricity_value = np.sqrt(1-(np.power(b,2)/np.power(a,2))) # 0: circle, 1: line
            
            # calculate smoothness
            contour_perimeter = cv2.arcLength(c, closed=True)
            ellipse_perimeter = np.pi*(3*(a+b)-np.sqrt((3*a+b)*(a+3*b)))
            smoothness_value = ellipse_perimeter/contour_perimeter # 0: bumpy, 1: circle
            
            contour_dataset['images'][date_number,y,x] = image_and_contour
            mask_dataset['images'][date_number,y,x] = np.repeat(mask[..., np.newaxis], 3, axis=2)
            mask_dataset['mask_num_pixel'][date_number,y,x] = np.sum(mask != 0)
            mask_dataset['circularity'][date_number,y,x] = calculate_circularity(mask != 0)
            mask_dataset['eccentricity'][date_number,y,x] = eccentricity_value
            mask_dataset['smoothness'][date_number,y,x] = smoothness_value
    
    mcam_data.save(contour_dataset, os.path.join(root_path, experiment, 'analysis', f'plate{plate_number+1}_mcam_contour_dataset.nc'),
                    include_timestamp=False, engine='ramona')
    mcam_data.save(mask_dataset, os.path.join(root_path, experiment, 'analysis', f'plate{plate_number+1}_mcam_mask_dataset.nc'),
                    include_timestamp=False, engine='ramona')


#%%
checker = np.full((2,2*len(dates_to_process)*8*6), False)


# %% get max projection images - FL
noise_location = np.load(os.path.join(root_path, 'noise_location.npy'))

for plate_number in range(2):
    i = 0
    
    mask_dataset = mcam_data.load(os.path.join(root_path, experiment, 'analysis',
                                               f'plate{plate_number+1}_mcam_mask_dataset.nc'), engine='ramona')
    
    if not any(checker[plate_number]):
        max_image_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process), image_shape=3072)
        exposure_time = np.full((len(dates_to_process),6*2,8), 0.) # switched!
        sum_of_signal = np.full((len(dates_to_process),6*2,8), 0.) # switched!
        sum_of_all = np.full((len(dates_to_process),6*2,8), 0.) # switched!
        exposure_time_dataarray = xr.DataArray(exposure_time,
                                               dims=['frame_number', 'image_y', 'image_x'],
                                               coords=[range(len(dates_to_process)), range(6*2), range(8)]) # switched!
        sum_of_signal_dataarray = xr.DataArray(sum_of_signal,
                                               dims=['frame_number', 'image_y', 'image_x'],
                                               coords=[range(len(dates_to_process)), range(6*2), range(8)]) # switched!
        sum_of_all_dataarray = xr.DataArray(sum_of_all,
                                            dims=['frame_number', 'image_y', 'image_x'],
                                            coords=[range(len(dates_to_process)), range(6*2), range(8)]) # switched!
        max_image_dataset['exposure_time'] = exposure_time_dataarray
        max_image_dataset['sum_of_signal'] = sum_of_signal_dataarray
        max_image_dataset['sum_of_all'] = sum_of_all_dataarray
        
    for plate_L_or_R in ['L', 'R']:
        
        for date_number, date_to_process in tqdm(enumerate(dates_to_process)):
            data = mcam_data.load(glob.glob(os.path.join(root_path, experiment,  date_to_process,
                                                          f'Day{date_number+1}_Plate{plate_number+1}_{plate_L_or_R}',
                                                          'center_FL_stack*', 'mcam_dataset.nc'))[0], engine='ramona')
            
            # load shift information
            shift_file_name = 'shift_info_20241016.json'
            with open(os.path.join(root_path, shift_file_name), 'r', encoding='utf-8') as f:
                shift_info = json.load(f)
            
            for y,x in np.ndindex(8,6):
                if checker[plate_number,i]:
                    i += 1
                    continue
                
                x_to_save = x+6 if plate_L_or_R == 'R' else x
                
                row_shift_per_stack = shift_info[f'({y},{x})']['row_shift_per_px']
                col_shift_per_stack = shift_info[f'({y},{x})']['col_shift_per_px']
                
                data_subset_gray_array = np.zeros((data.images.shape[0], 3072, 3072), dtype=np.uint8)
                
                # remove hot pixels
                for z_stack in range(data.images.shape[0]):
                    data_bayer = data.images[z_stack,y,x].values
                    data_bayer[::2,1::2] = 0 # blue
                    data_bayer[1::2,::2] = 0 # red
                    data_bayer[noise_location[y,x]] = 0 # noise
                    data_subset_gray_array[z_stack] = cv2.cvtColor(data_bayer,
                                                                   cv2.COLOR_BayerGR2RGB)[...,1]

                for z_stack in range(data.images.shape[0]):
                    data_subset_gray_array[z_stack] = np.roll(data_subset_gray_array[z_stack],
                                                     (int(row_shift_per_stack*z_stack),
                                                      int(col_shift_per_stack*z_stack)),
                                                     axis=(0,1))
                data_subset_gray_array_max_projection = np.max(data_subset_gray_array, axis=0)
                data_subset_gray_array_max_projection = np.repeat(data_subset_gray_array_max_projection[..., np.newaxis], 3, axis=2)
                data_subset_gray_array_max_projection[...,0] = 0
                data_subset_gray_array_max_projection[...,2] = 0
                # crop background
                background = mask_dataset.images[date_number,x_to_save,y].values != 255 # switched!
                data_subset_gray_array_max_projection[background] = 0
                max_image_dataset['images'][date_number,x_to_save,y] = data_subset_gray_array_max_projection # switched!
                max_image_dataset['exposure_time'][date_number,x_to_save,y] = data.exposure.data[y,x] # switched!
                max_image_dataset['sum_of_signal'][date_number,x_to_save,y] = np.sum(data_subset_gray_array_max_projection) # switched!
                sum_of_all_to_save = mask_dataset.images[date_number,x_to_save,y].values
                max_image_dataset['sum_of_all'][date_number,x_to_save,y] = np.sum(sum_of_all_to_save[...,1]) # switched!
                checker[plate_number,i] = True
                i += 1
    mcam_data.save(max_image_dataset, os.path.join(root_path, experiment, 'analysis', f'plate{plate_number+1}_mcam_max_image_FL_dataset.nc'),
                    include_timestamp=False, engine='ramona')