# -*- coding: utf-8 -*-
"""
The 'arunet_olt' model was trained on metrics derived from high-density LiDAR data to segment logging trails in forest.
This model takes a 256 by 256 input image, including canopy height models (CHM), digital surface models (DSM), and digital 
elevation models (DEM), with a cell size of 0.4 meters. The output is an image that includes the segmented logging trails, polygons of OLTs and the centerlines.
Created on Tue Mar 21 10:09:23 2023
@author: Omid Abdi
"""
#1. Import required python packages
#Import required python packages
import numpy as np
from tensorflow.keras.models import load_model
from skimage import io
import os
import rasterio#'pip install rasterio' in Python Command Prompt 
from rasterio.features import shapes
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry import LineString, Polygon, Point
from centerline.geometry import Centerline#pip
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from rasterio.merge import merge

import sys
print(sys.executable)


#2. Load the trained arunet model
## Load the trained arunet model i.e 'raunet_olt.h5'
olt_model = load_model(r'C:\OLTs\arunet_olt\model\ARUNet\AttResUnet_olt', custom_objects={"K": K})

#3. Load the input image patches
# Load the input patches of the images and normalize them
image_directory =r'C:\OLTs\arunet_olt\data\patches\images\*.tif'
SIZE = 256
image_dataset = []
images=io.imread_collection(image_directory,plugin='tifffile')
for j in range(len(images)):
        #print(j)
    # j=frangi(images[j],black_ridges=False)
    # x.append(j)
    #image_dataset.append(images[j])
    image_dataset.append(images[j].reshape(256,256,3))
image_dataset = np.array(image_dataset)
image_dataset.sort()
image_dataset = np.array(image_dataset)/255.
#image_dataset = np.transpose(image_dataset, (0, 2, 3, 1)) 

#4. Make prediction of OLTs
# Make predictions on the input images using the trained arunet model
olt_prediction = olt_model.predict(image_dataset)

#5. Define the output directory
# Define the output directory path as 'output_olt'
output_dir = os.path.expanduser(r'C:\OLTs\arunet_olt\output')

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Save the predicted outputs with coordinate systems as their corresponding inputs
output_files = []
for i in range(len(olt_prediction)):
    filename = os.path.basename(images.files[i])
    output_path = os.path.join(filename)
    
    with rasterio.open(images.files[i]) as src:
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw')
        
        with rasterio.open(output_path, 'w+', **profile) as dst:
            dst.write(np.squeeze(olt_prediction[i]), 1)
            output_files.append(output_path)

            
# Mosaic the outputs into a single raster using the "mean" operator
mosaic_output_path = os.path.join(output_dir, "mosaic_olt.tif")
src_files_to_mosaic = [rasterio.open(fp) for fp in output_files]

# Perform the mosaicking
mosaic, mosaic_transform = merge(src_files_to_mosaic, method='last')


# Save the mosaicked raster
mosaic_profile = src_files_to_mosaic[0].profile
mosaic_profile.update({
    'height': mosaic.shape[1],
    'width': mosaic.shape[2],
    'transform': mosaic_transform,
    'compress': 'lzw'
})

# Write the 2D data from the mosaic array
with rasterio.open(mosaic_output_path, 'w', **mosaic_profile) as mosaic_dst:
    mosaic_dst.write(mosaic[0].astype(rasterio.float32), 1)  # Use mosaic[0] to extract the 2D array

# Close all the source files
for src in src_files_to_mosaic:
    src.close()

print(f"Mosaicked raster saved to {mosaic_output_path}")

# Process the mosaicked raster: Apply threshold
with rasterio.open(mosaic_output_path) as src:
    mosaic_data = src.read(1)  # Read the first (and only) band
    mosaic_profile = src.profile

# Apply threshold: Assign 1 to pixels greater than 0.9, and 0 otherwise
thresholded_pixels = (mosaic_data > 0.95).astype(np.uint8)

# Save the thresholded raster
thresholded_output_path = mosaic_output_path.replace(".tif", "_thresholded.tif")
mosaic_profile.update(dtype=rasterio.uint8, compress='lzw')

with rasterio.open(thresholded_output_path, 'w', **mosaic_profile) as dst:
    dst.write(thresholded_pixels, 1)

print(f"Thresholded raster saved to {thresholded_output_path}")


#Step 6: Function to convert raster to smoothed polygons and filter small polygons
def raster_to_smoothed_polygons(binary_raster_path, output_dir, tolerance=0.75, min_area=20):
    """
    Converts a binary raster to smoothed polygons and removes small polygons.
    """
    with rasterio.open(binary_raster_path) as src:
        labeled_array = src.read(1)
        transform = src.transform
        crs = src.crs

        # Mask to exclude 0 values
        mask = labeled_array.astype(np.uint8)

        smoothed_polygons = []
        for shape, value in shapes(labeled_array, mask=mask, transform=transform):
            if value != 0:  # Skip background values
                geom = shape["coordinates"]

                if len(geom) == 1:  # Single polygon
                    polygon = Polygon(geom[0])
                else:  # Polygon with holes
                    exterior = geom[0]
                    interiors = geom[1:]
                    polygon = Polygon(exterior, interiors)

                # Simplify polygon to smooth boundaries
                simplified_polygon = polygon.simplify(tolerance=tolerance, preserve_topology=True)

                # Add only if the polygon's area exceeds the minimum threshold
                if simplified_polygon.area > min_area:
                    smoothed_polygons.append(simplified_polygon)

        # Create GeoDataFrame for the smoothed polygons
        gdf_smoothed_polygons = gpd.GeoDataFrame(geometry=smoothed_polygons, crs=crs)

        # Save polygons to shapefile
        polygons_shp = os.path.join(output_dir, 'sm_olt.shp')
        gdf_smoothed_polygons.to_file(polygons_shp)

        return polygons_shp

# Convert the binary raster to smoothed polygons
smoothed_polygons_path = raster_to_smoothed_polygons(thresholded_output_path, output_dir)
gdf_smoothed_polygons = gpd.read_file(smoothed_polygons_path)
print(f"Smoothed polygons saved to {smoothed_polygons_path}")

#Step 5: Create centerlines from polygons
centerlines = []
for polygon in gdf_smoothed_polygons.geometry:
    attributes = {"id": 1, "name": "centerline", "valid": True}
    centerline = Centerline(polygon, **attributes)
    centerlines.append(centerline.geometry)
# Create a GeoDataFrame from the centerlines
gdf_centerlines = gpd.GeoDataFrame(geometry=centerlines, crs=src.crs)

# Plot the centerlines
plt.figure(figsize=(10, 10))
gdf_centerlines.plot()
plt.title('Centerlines from Polygons')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Save the centerlines as a shapefile
output_centerlines = os.path.join(output_dir, 'centerlines.shp')
gdf_centerlines.to_file(output_centerlines)
print(f"Centerlines saved to {output_centerlines}")

# Print some of the centerline results
print(gdf_centerlines.head())

