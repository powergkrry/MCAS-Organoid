# cellpose version: 3.1.1.1

import tifffile as tiff
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cellpose import models
import pandas as pd


#%%
root_path = '/media/kanghyun/Sunday/Temporary/kanghyun/Eagle'
model = models.Cellpose(model_type='cyto2', gpu=True)
channels = [[0,0]]
num_of_cells = np.full((2,6*2,8), 0)

#%%
for plate_L_or_R in ['L', 'R']:
    for y,x in tqdm(np.ndindex(8,6)):
        if plate_L_or_R == 'R':
            x_to_save = x+6
            BF_image_path = os.path.join(root_path, '86_N2a_GFP_Plate1_R_BF_inverted',
                                         '05_Normalized_FOV_tiled_data', 'focal_stack_0',
                                         f'N2a_GFP_Plate1_R_BF_inverted_cam_y{y:02d}_x{x:02d}.tif')
            FL_image_path = os.path.join(root_path, '87_N2a_GFP_Plate1_R_FL_inverted',
                                         '05_Normalized_FOV_tiled_data', 'focal_stack_0',
                                         f'N2a_GFP_Plate1_R_FL_inverted_cam_y{y:02d}_x{x:02d}.tif')
        else:
            x_to_save = x
            BF_image_path = os.path.join(root_path, '84_N2a_GFP_Plate1_L_BF_inverted',
                                         '05_Normalized_FOV_tiled_data', 'focal_stack_0',
                                         f'N2a_GFP_Plate1_L_BF_inverted_cam_y{y:02d}_x{x:02d}.tif')
            FL_image_path = os.path.join(root_path, '85_N2a_GFP_Plate1_L_FL_inverted',
                                         '05_Normalized_FOV_tiled_data', 'focal_stack_0',
                                         f'N2a_GFP_Plate1_L_FL_inverted_cam_y{y:02d}_x{x:02d}.tif')

        BF_image = tiff.imread(BF_image_path)[...,1]
        FL_image = tiff.imread(FL_image_path)[...,1]
        
        BF_masks, flows, styles, diams = model.eval(BF_image, diameter=None, channels=channels)
        FL_masks, flows, styles, diams = model.eval(FL_image, diameter=None, channels=channels)

        # save masks
        temp_path = os.path.join(root_path, '84_N2a_GFP_Plate1_L_BF_inverted', '99_Other_unorganized')
        # specifying L and R makes easier to use later. less confusing
        if plate_L_or_R == 'R':
            np.save(os.path.join(temp_path,
                                 f'N2a_GFP_Plate1_R_BF_masks_inverted_cam_y{y:02d}_x{x:02d}.npy'), BF_masks)
            np.save(os.path.join(temp_path,
                                 f'N2a_GFP_Plate1_R_FL_masks_inverted_cam_y{y:02d}_x{x:02d}.npy'), FL_masks)
        else:
            np.save(os.path.join(temp_path,
                                 f'N2a_GFP_Plate1_L_BF_masks_inverted_cam_y{y:02d}_x{x:02d}.npy'), BF_masks)
            np.save(os.path.join(temp_path,
                                 f'N2a_GFP_Plate1_L_FL_masks_inverted_cam_y{y:02d}_x{x:02d}.npy'), FL_masks)
        
        # count # of cells
        num_of_cells[0,x_to_save,y] = len(pd.unique(np.ravel(FL_masks)))
        num_of_cells[1,x_to_save,y] = len(pd.unique(np.ravel(BF_masks)))
np.save(os.path.join(temp_path, 'num_of_cells.npy'), num_of_cells)