import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

import geopandas as gpd
import rasterio
from collections import Counter
import shapely.geometry
from descartes import PolygonPatch
import cv2
from geopandas.geoseries import *
import h5py # just a safety check so the checkpoint callback doesnt crash
from scipy.misc import imresize
from shapely.ops import unary_union

from matplotlib.collections import PatchCollection

# Recycle a couple of functions from previous notebooks...
def polycoords(poly):
    """Convert a polygon into the format expected by OpenCV
    """
    if poly.type in ['MultiPolygon', 'GeometryCollection']:
        return [np.array(p.exterior.coords) for p in poly if p.type == 'Polygon']
    elif poly.type == 'Polygon':
        return [np.array(poly.exterior.coords)]
    else:
        print('Encountered unrecognized geometry type {}. Ignoring.'.format(poly.type))
        return []

    
def make_mask(img_shape, poly):
    """Make a mask from a polygon"""
    poly_pts = polycoords(poly)
    polys = [x.astype(int) for x in poly_pts]
    # Create an empty mask and then fill in the polygons
    mask = np.zeros(img_shape[:2])
    cv2.fillPoly(mask, polys, 255)
    return mask.astype('uint8')


def scale_bands(img, lower_pct = 1, upper_pct = 99):
    """Rescale the bands of a multichannel image for display"""
    img_scaled = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[2]):
        band = img[:, :, i]
        lower, upper = np.percentile(band, [lower_pct, upper_pct])
        band = (band - lower) / (upper - lower) * 255
        img_scaled[:, :, i] = np.clip(band, 0, 255).astype(np.uint8)
    return img_scaled


def resize(img, new_shape):
    img_resized = np.zeros(new_shape+(img.shape[2],)).astype('float32')
    for i in range(img.shape[2]):
        img_resized[:, :, i] = imresize(img[:, :, i], new_shape, interp='bicubic')
    return img_resized


# Build a training set
def make_set(image_summary, vectors, training_set_size, validation_set_size, input_size, random=np.random):
    rows_to_use = random.choice(image_summary.index, training_set_size + validation_set_size, replace=False)
    
    train_rows_to_use = rows_to_use[:training_set_size]
    val_rows_to_use = rows_to_use[training_set_size:]
    
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    
    
    # make training set
    for i, row in image_summary.loc[train_rows_to_use].iterrows():
        with rasterio.open(row.image_name) as src:
            img = src.read().transpose([1,2,0])
#             img = resize(img, (input_size, input_size))
            # img = scale_bands(img, lower_pct = 5, upper_pct = 95)
            img_bounds = shapely.geometry.box(*src.bounds)
            img_transform = list(np.array(~src.affine)[[0, 1, 3, 4, 2, 5]])
           
        masks = []
        for poly in vectors:
            # Get the intersection between the polygon and the image bounds
            mask_poly = poly.intersection(img_bounds)

            # Transform it into pixel coordinates
            mask_poly_pxcoords = shapely.affinity.affine_transform(mask_poly, img_transform)

            # Convert the polygon into a mask
            mask = make_mask(img.shape[:2], mask_poly_pxcoords)
            mask = imresize(mask, (INPUT_SIZE, INPUT_SIZE))
            
            masks.append(mask[..., None])
        masks = np.concatenate(masks, axis=2)
        img = resize(img, (input_size, input_size))
        
        # Add each mask to a list
        X_train.append(img[None, ...]) # These need to be truncated for U-Net
        Y_train.append(masks[None, ...])
        
    # make validation set
    for i, row in image_summary.loc[val_rows_to_use].iterrows():
        with rasterio.open(row.image_name) as src:
            img = src.read().transpose([1,2,0])
#             img = resize(img, (input_size, input_size))
            # img = scale_bands(img, lower_pct = 5, upper_pct = 95)
            img_bounds = shapely.geometry.box(*src.bounds)
            img_transform = list(np.array(~src.affine)[[0, 1, 3, 4, 2, 5]])
           
        masks = []
        for poly in vectors:
            # Get the intersection between the polygon and the image bounds
            mask_poly = poly.intersection(img_bounds)

            # Transform it into pixel coordinates
            mask_poly_pxcoords = shapely.affinity.affine_transform(mask_poly, img_transform)

            # Convert the polygon into a mask
            mask = make_mask(img.shape[:2], mask_poly_pxcoords)
            mask = imresize(mask, (INPUT_SIZE, INPUT_SIZE))
            
            masks.append(mask[..., None])
        masks = np.concatenate(masks, axis=2)
        img = resize(img, (input_size, input_size))
        
        # Add each mask to a list
        X_val.append(img[None, ...]) # These need to be truncated for U-Net
        Y_val.append(masks[None, ...])

            
    # Concatenate the results
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    Y_val = np.concatenate(Y_val, axis=0)
    
    # Normalize the values
    X_train = X_train.astype('float32')
    X_train = (X_train / X_train.max() - 0.5) * 2 # put X in range [-1, 1]
    Y_train = Y_train.astype('float32') / 255 # put Y in range [0, 1]
    X_val = X_val.astype('float32')
    X_val = (X_val / X_val.max() - 0.5) * 2 # put X in range [-1, 1]
    Y_val = Y_val.astype('float32') / 255 # put Y in range [0, 1]
    
    return X_train, Y_train, X_val, Y_val, train_rows_to_use, val_rows_to_use

sea = gpd.read_file('../AOI_4_Shanghai_Train/geojson/shanghai_coastal_waters.geojson')
sea.head()
water_files = ! ls ../AOI_4_Shanghai_Train/line_water_geojson_full/
print('There are {} waterway shapefiles'.format(len(water_files)))
waterways = gpd.read_file('../AOI_4_Shanghai_Train/line_water_geojson_full/{}'.format(water_files[1]))
waterways.head()
# load-in rest of waterways shapefiles
for file in water_files[2:]:
    waterways = waterways.append(gpd.read_file('../AOI_4_Shanghai_Train/line_water_geojson_full/{}'.format(file)), ignore_index=True)


# clean up dataframe
waterways.drop(['4297_poly','id','polyon'], axis=1, inplace=True)
# view data frame
print(waterways.shape)
waterways.head()
# load-in summary data of satellite images
image_summary = gpd.read_file('../AOI_4_Shanghai_Train/vectors/image_summary.geojson')


part1 = sea.unary_union
part2 = unary_union([x.buffer(0) for x in waterways.geometry])
all_water = part1.union(part2)

images_containing_water = image_summary[image_summary.intersects(all_water)]
images_containing_water.head()
images_containing_water.shape

file_name = images_containing_water.image_name.values[2]

with rasterio.open(file_name) as src:
        img = scale_bands(src.read([5, 3, 2]).transpose([1,2,0]))
        img_bounds = shapely.geometry.box(*src.bounds)
        img_transform = list(np.array(~src.affine)[[0, 1, 3, 4, 2, 5]])
        
# Get the intersection between the forest and the image bounds
image_water_area = all_water.intersection(img_bounds)

# Transform it into pixel coordinates
image_water_area_pxcoords = shapely.affinity.affine_transform(image_water_area, img_transform)


vectors = [all_water]

INPUT_SIZE = 240
random = np.random.RandomState(2) 
X_train, Y_train, X_val, Y_val, train_rows, val_rows = make_set(image_summary, 
                                                                vectors, 
                                                                4000, 
                                                                500, 
                                                                INPUT_SIZE, 
                                                                random)

# set, random state, size, mark
pickle.dump(X_train, open('../pickle_jar/X_train_r2_4000_V.p','wb'),protocol=4)
pickle.dump(Y_train, open('../pickle_jar/Y_train_r2_4000_V.p','wb'),protocol=4)
pickle.dump(X_val, open('../pickle_jar/X_val_r2_500_III.p','wb'),protocol=4)
pickle.dump(Y_val, open('../pickle_jar/Y_val_r2_500_III.p','wb'),protocol=4)

for i in range(Y_train.shape[0]):
    x = X_train[i]
    y = Y_train[i]

    # Pick out which target to look at
    CLASS_NO = 0
    targ = y[:, :, CLASS_NO]

    # Plot it
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
    ax1.imshow(scale_bands(x[:,:,[4,2,1]])) # This index starts at 0, so I had to decrement
    ax2.imshow(targ, vmin=0, vmax=1)

    ax1.set_title('Image')
    ax2.set_title('Ground Truth');
    ax1.grid();
    ax2.grid();
    plt.show()
    
    print('{}/{}'.format(i+1, Y_train.shape[0]))
    
    if i == 7:
        break
    
    time.sleep(1)
    display.clear_output(wait=True)
    