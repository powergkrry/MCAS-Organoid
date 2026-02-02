# sam2 version: 0.4.1
import json
import numpy as np
import os
import cv2
import xarray as xr
import glob
import copy
from owl.mcam_data import new_dataset, new_rgb_dataset
from owl import mcam_data
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from owl.instruments.falcon import alloc_buffer_for_acquisition
from skimage.measure import perimeter
from skimage.measure import EllipseModel


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def new_dataset_with_frame_number(z_num_steps):
    y_dim = 8
    x_dim = 6
    image_shape = 3072
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

def new_rgb_dataset_with_frame_number(z_num_steps):
    y_dim = 6*2 # 8 # switched!
    x_dim = 8 # 6*2 # switched!
    image_shape = 3072
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
root_path = '/media/kanghyun/Sunday/Temporary/kanghyun/Eagle/79_UNC_first_try'
dates_to_process = ['20240207', '20240208', '20240209', '20240210', '20240211', '20240212', '20240213', '20240214',
                    '20240215', '20240216', '20240217', '20240218', '20240219', '20240220', '20240221', '20240222', '20240223']
experiment = 'KH_C'
if not os.path.exists(os.path.join(root_path, experiment, 'analysis')):
    os.makedirs(os.path.join(root_path, experiment, 'analysis'))

sam_checkpoint = os.path.join(root_path, 'sam_vit_h_4b8939.pth')
sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)


# %% get max projection images, segmentation maps, and images with contours
max_image_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process))
contour_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process))
mask_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process))

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

for plate_L_or_R in ['L', 'R']:
    for date_number, date_to_process in tqdm(enumerate(dates_to_process)):
        # load shift information
        if date_to_process < '20240213':
            shift_file_name = 'shift_info_20240124.json'
        else:
            shift_file_name = 'shift_info_20240213.json'
        with open(os.path.join(root_path, shift_file_name), 'r', encoding='utf-8') as f:
            shift_info = json.load(f)
        
        # load data to segment
        try:
            data = mcam_data.load(glob.glob(os.path.join(root_path, experiment,  date_to_process,
                                                          f'{experiment}_{plate_L_or_R}_stack*',
                                                          'mcam_dataset.nc'))[0], engine='ramona')
        except:
            data = mcam_data.load(glob.glob(os.path.join(root_path, experiment,  date_to_process,
                                                          f'{experiment}_{plate_L_or_R}1_stack*',
                                                          'mcam_dataset.nc'))[0], engine='ramona')
        
        # load base information
        with open(os.path.join(root_path, experiment, date_to_process,
                               f'base_info_{plate_L_or_R}.json'), 'r', encoding='utf-8') as f:
            base_info = json.load(f)
        
        for y,x in np.ndindex(8,6):
            
            x_to_save = x+6 if plate_L_or_R == 'R' else x
                        
            row_shift_per_stack = shift_info[f'({y},{x})']['row_shift_per_px']
            col_shift_per_stack = shift_info[f'({y},{x})']['col_shift_per_px']
            start_to_use = base_info[f'({y},{x})']['start_to_use']
            end_to_use = base_info[f'({y},{x})']['end_to_use']+1
            data_subset_gray_array = np.zeros((end_to_use-start_to_use, 3072, 3072), dtype=np.uint8)
            
            data_subset = data.images[start_to_use:end_to_use,y,x]
            for z_stack in range(end_to_use-start_to_use):
                data_subset_gray_array[z_stack] = cv2.cvtColor(data_subset[z_stack].values,
                                                               cv2.COLOR_BayerGR2GRAY)
                data_subset_gray_array[z_stack] = np.roll(data_subset_gray_array[z_stack],
                                                 (int(row_shift_per_stack*z_stack),
                                                  int(col_shift_per_stack*z_stack)),
                                                 axis=(0,1))
            data_subset_gray_array_max_projection = np.max(data_subset_gray_array, axis=0)
            data_subset_gray_array_max_projection = np.repeat(data_subset_gray_array_max_projection[..., np.newaxis], 3, axis=2)
            max_image_dataset['images'][date_number,x_to_save,y] = data_subset_gray_array_max_projection # switched!
            
            predictor.set_image(data_subset_gray_array_max_projection)
            
           # load negative location
            try:
                exclude_input_point_from_json = base_info[f'({y},{x})']['exclude_position']
            except:
                # load positive location
                input_point_from_json = base_info[f'({y},{x})']['center_organoid_position']
                if np.array(input_point_from_json).ndim == 1:
                    input_point_from_json = [input_point_from_json]
                input_point = np.array(input_point_from_json)
                input_label = np.array([1]*len(input_point_from_json))
            else:
                if np.array(exclude_input_point_from_json).ndim == 1:
                    exclude_input_point_from_json = [exclude_input_point_from_json]
                input_point = np.concatenate((input_point_from_json, exclude_input_point_from_json))
                input_label = np.array([1]*len(input_point_from_json) + [0]*len(exclude_input_point_from_json))
            
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False)
            
            # fill holes
            mask = (masks[0]*255).astype(np.uint8)
            contour,hier = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                cv2.drawContours(mask,[cnt],0,255,-1)
            
            # smoothing
            blur = cv2.GaussianBlur(mask, (15,15), 0)
            mask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[-1]
            
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
            smoothness_value = ellipse_perimeter/contour_perimeter
            
            contour_dataset['images'][date_number,x_to_save,y] = image_and_contour # switched!
            mask_dataset['images'][date_number,x_to_save,y] = np.repeat(mask[..., np.newaxis], 3, axis=2) # switched!
            mask_dataset['mask_num_pixel'][date_number,x_to_save,y] = np.sum(mask != 0) # switched!
            mask_dataset['circularity'][date_number,x_to_save,y] = calculate_circularity(mask != 0) # switched!
            mask_dataset['eccentricity'][date_number,x_to_save,y] = eccentricity_value # switched!
            mask_dataset['smoothness'][date_number,x_to_save,y] = smoothness_value # switched!
mcam_data.save(max_image_dataset, os.path.join(root_path, experiment, 'analysis', 'mcam_max_image_dataset.nc'),
                include_timestamp=False, engine='ramona')
mcam_data.save(contour_dataset, os.path.join(root_path, experiment, 'analysis', 'mcam_contour_dataset.nc'),
                include_timestamp=False, engine='ramona')
mcam_data.save(mask_dataset, os.path.join(root_path, experiment, 'analysis', 'mcam_mask_dataset.nc'),
                include_timestamp=False, engine='ramona')
