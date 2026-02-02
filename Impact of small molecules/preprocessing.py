# transformers version: 4.56.1
import json
import numpy as np
import os
import cv2
import xarray as xr
import glob
import copy
from owl.mcam_data import new_rgb_dataset
from owl.stitch import hugin_stitching
from owl import mcam_data
from tqdm import tqdm
from owl.instruments.falcon import alloc_buffer_for_acquisition
from skimage.measure import perimeter
from skimage.measure import EllipseModel
import tifffile as tiff
import torch
from torch import nn
from transformers import SamModel
from transformers import SamProcessor


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

def crop_center(img, crop_size_y, crop_size_x):
    y, x, _ = img.shape
    starty = y // 2 - (crop_size_y // 2)
    startx = x // 2 - (crop_size_x // 2)
    return img[starty : starty + crop_size_y, startx : startx + crop_size_x]


#%%
root_path = '/media/kanghyun/Sunday/Temporary/kanghyun/Eagle/81_UNC_multiple_drugs'
dates_to_process = ['20240517', '20240518', '20240519', '20240520', '20240521', '20240522', '20240523', '20240524',
                    '20240525', '20240526', '20240527', '20240528', '20240529', '20240530', '20240531']
experiment = 'KH_A'
if not os.path.exists(os.path.join(root_path, experiment, 'analysis')):
    os.makedirs(os.path.join(root_path, experiment, 'analysis'))
sam = SamModel.from_pretrained("facebook/sam-vit-huge")
# sam.load_state_dict(torch.load(sam_checkpoint))
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
# make sure we only compute gradients for mask decoder
for name, param in sam.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")


#%%
checker = np.full((2,2*len(dates_to_process)*8*6), False) # NAS somehow sometimes stops..


# %% get max projection images, segmentation maps, and images with contours - 4 images stiching
for plate_number in range(2):
    i = 0
    
    if not any(checker[plate_number]):
        max_image_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process), image_shape=3584)
        contour_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process), image_shape=3584)
        mask_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process), image_shape=3584)
        mask_num_pixel = np.full((len(dates_to_process),6*2,8), 0) # switched!
        circularity = np.full((len(dates_to_process),6*2,8), 0.) # switched!
        mask_num_pixel_dataarray = xr.DataArray(mask_num_pixel,
                                                dims=['frame_number', 'image_y', 'image_x'],
                                                coords=[range(len(dates_to_process)), range(6*2), range(8)]) # switched!
        circularity_dataarray = xr.DataArray(circularity,
                                              dims=['frame_number', 'image_y', 'image_x'],
                                              coords=[range(len(dates_to_process)), range(6*2), range(8)]) # switched!
        mask_dataset['mask_num_pixel'] = mask_num_pixel_dataarray
        mask_dataset['circularity'] = circularity_dataarray
    
    for plate_L_or_R in ['L', 'R']:
        for date_number, date_to_process in tqdm(enumerate(dates_to_process)):
            # load shift information
            if date_to_process < '20240213':
                shift_file_name = 'shift_info_20240124.json'
            elif date_to_process >= '20240213' and date_to_process < '20240517':
                shift_file_name = 'shift_info_20240213.json'
            else:
                shift_file_name = 'shift_info_20241016.json'
            with open(os.path.join(root_path, shift_file_name), 'r', encoding='utf-8') as f:
                shift_info = json.load(f)
                
            # load base information
            with open(os.path.join(root_path, experiment, date_to_process,
                                   f'plate{plate_number+1}_base_info_stitched_{plate_L_or_R}.json'), 'r', encoding='utf-8') as f:
                base_info = json.load(f)
            
            for y,x in np.ndindex(8,6):
                if checker[plate_number,i]:
                    i += 1
                    continue
                print(y,x)
                
                x_to_save = x+6 if plate_L_or_R == 'R' else x
                
                row_shift_per_stack = shift_info[f'({y},{x})']['row_shift_per_px']
                col_shift_per_stack = shift_info[f'({y},{x})']['col_shift_per_px']
                start_to_use = base_info[f'({y},{x})']['start_to_use']
                end_to_use = base_info[f'({y},{x})']['end_to_use']+1
                
                # load 4 images, max projection 4 images, stitch 4 images, and SAM
                single_image_dataset = new_rgb_dataset(N_cameras=(2,2), image_shape=(3072,3072,3))
                for scan_y, scan_x in np.ndindex(2,2):
                    data = mcam_data.load(glob.glob(os.path.join(root_path, experiment,  date_to_process,
                                                                 f'*_Plate{plate_number+1}_{plate_L_or_R}', f'FOV_y{scan_y:02d}_x{scan_x:02d}*',
                                                                 'mcam_dataset.nc'))[0], engine='ramona') # multi drugs KH_A
                
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
                    single_image_dataset['images'][scan_y, scan_x] = data_subset_gray_array_max_projection
                mcam_data.save(single_image_dataset, os.path.join(root_path, experiment, 'analysis', 'temp.nc'),
                               include_timestamp=False, engine='ramona')
                try:
                    stitch_path, pto_path = hugin_stitching(os.path.join(root_path, experiment, 'analysis', 'temp.nc'),
                                                            ignore_calibration=False, attempt_custom_alignment=True)
                except:
                    stitch_path, pto_path = hugin_stitching(os.path.join(root_path, experiment, 'analysis', 'temp.nc'),
                                                            estimated_overlap=(0.33,0.33),
                                                            ignore_calibration=False, attempt_custom_alignment=True)
                tiled_image = tiff.imread(stitch_path)[...,:3]
                crop_image_size = 3584
                input_box = np.array([0, 0, crop_image_size, crop_image_size])
                image_shape = tiled_image.shape[:2]
                before_center = np.array([0, 0])
                current_center = np.array([image_shape[0]//2, image_shape[1]//2])
                patch_center = np.array([crop_image_size//2, crop_image_size//2])
                iter_num = 0
                mask = 0
                while (np.sum(mask != 0) < 100000 or np.sqrt(np.sum(np.power(before_center-current_center, 2))) > 50) and iter_num < 7:
                    data_subset_gray_array_max_projection = tiled_image[current_center[0]-crop_image_size//2:current_center[0]+crop_image_size//2,
                                                                        current_center[1]-crop_image_size//2:current_center[1]+crop_image_size//2]
                    # ValueError: could not broadcast input array from shape (3584,0,3) into shape (3584,3584,3)
                    # then just restart with different random center point
                    # if iter_num >= 7 it will just give up and move next
                    try:
                        max_image_dataset['images'][date_number,x_to_save,y] = data_subset_gray_array_max_projection # switched!
                    except:
                        print('ValueError')
                        before_center = np.array([0, 0])
                        current_center = np.array([image_shape[0]//2, image_shape[1]//2])
                        continue
                    
                    prompt = [crop_image_size//2, crop_image_size//2]
                    inputs = {
                        k: v.squeeze(0) 
                        for k, v in processor(data_subset_gray_array_max_projection, input_points=[[prompt]], return_tensors="pt").items()
                        }
                    outputs = sam(pixel_values=inputs["pixel_values"][None,...].to(device),
                                  input_points=inputs["input_points"][None,...].to(device),
                                  multimask_output=False)
                    mask = outputs.pred_masks.squeeze(1)
                    mask = nn.functional.interpolate(mask, size=(crop_image_size, crop_image_size),
                                                     mode='bilinear', align_corners=False)
                    mask = mask > 0.0
                    
                    # fill holes
                    mask = (mask[0,0].cpu().detach().numpy()*255).astype(np.uint8)
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
                    
                    # get center of contour
                    M = cv2.moments(c)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                    
                    before_center = current_center
                    current_patch_center = np.array([cy, cx])
                    current_center = before_center + (current_patch_center - patch_center)
                    # when segmentation mask is too small, I know it is wrong
                    if np.sum(mask != 0) < 100000:
                        input_point = [prompt] + np.random.uniform(-100, 100, (1,2)).astype(int)
                    else: # by resetting this, I can at least make sure that segmentation was performed based on the center point
                        input_point = np.array([[crop_image_size//2, crop_image_size//2]])
                    iter_num += 1
                
                contour_dataset['images'][date_number,x_to_save,y] = image_and_contour # switched!
                mask_dataset['images'][date_number,x_to_save,y] = np.repeat(mask[..., np.newaxis], 3, axis=2) # switched!
                mask_dataset['mask_num_pixel'][date_number,x_to_save,y] = np.sum(mask != 0) # switched!
                mask_dataset['circularity'][date_number,x_to_save,y] = calculate_circularity(mask != 0) # switched!
                
                delete_path = os.path.join(root_path, experiment, 'analysis', 'temp.nc')
                os.system(f'rm {delete_path}')
                checker[plate_number,i] = True
                i += 1
    mcam_data.save(max_image_dataset, os.path.join(root_path, experiment, 'analysis', f'plate{plate_number+1}_stitched_mcam_max_image_dataset.nc'),
                    include_timestamp=False, engine='ramona')
    mcam_data.save(contour_dataset, os.path.join(root_path, experiment, 'analysis', f'plate{plate_number+1}_stitched_mcam_contour_dataset.nc'),
                    include_timestamp=False, engine='ramona')
    mcam_data.save(mask_dataset, os.path.join(root_path, experiment, 'analysis', f'plate{plate_number+1}_stitched_mcam_mask_dataset.nc'),
                    include_timestamp=False, engine='ramona')

delete_path = os.path.join(root_path, experiment, 'analysis', 'temp*')
os.system(f'rm -r {delete_path}')


#%% generate .nc for manually annotated data
for plate_number in range(2):
    max_image_dataset = mcam_data.load(os.path.join(root_path, experiment, 'analysis',
                                                    f'plate{plate_number+1}_stitched_mcam_max_image_dataset.nc'), engine='ramona')
    contour_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process), image_shape=3584)
    mask_dataset = new_rgb_dataset_with_frame_number(len(dates_to_process), image_shape=3584)
    
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
    
    mcam_data.save(contour_dataset, os.path.join(root_path, experiment, 'analysis', f'plate{plate_number+1}_stitched_mcam_contour_dataset.nc'),
                    include_timestamp=False, engine='ramona')
    mcam_data.save(mask_dataset, os.path.join(root_path, experiment, 'analysis', f'plate{plate_number+1}_stitched_mcam_mask_dataset.nc'),
                    include_timestamp=False, engine='ramona')
