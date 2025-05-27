import os
import re
import numpy as np
import tifffile
from utils import filemanager as fm

# Read path
image_folder = '/media/mkhatereh/Elements/RSDE/S2_DL/Crop_Germany/Data/Images'
mask_folder = '/media/mkhatereh/Elements/RSDE/S2_DL/Crop_Germany/Data/Masks_buffer'
output_image_folder = '/media/mkhatereh/Elements/RSDE/S2_DL/Crop_Germany/Data/Images_clip_256_buffer'
output_mask_folder = '/media/mkhatereh/Elements/RSDE/S2_DL/Crop_Germany/Data/Masks_clip_256_buffer'
clip_size = 256

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

#Get all mean composite image files
image_files = [f for f in os.listdir(image_folder) if f.endswith('_mean.tif')]

# Group by area ID
# Pattern: S-2_composites_Baden-Württemberg_2020_1_2020-02_mean.tif
area_groups = {}
pattern = r"S-2_composites_(.+?)_(\d{4})_(\d+)_\d{4}-(\d{2})_mean\.tif"

for f in image_files:
    match = re.match(pattern, f)
    if match:
        region = match.group(1)
        year = match.group(2)
        area_idx = match.group(3)  # area number: 110
        month = match.group(4)
        area_key = f"{region}_{year}_{area_idx}"  # e.g., Baden-Württemberg_2020_3
        area_groups.setdefault(area_key, []).append(f)

# Process each area 
for area_key, files in area_groups.items():
    # Sort by temporal month (to keep correct order)
    files = sorted(files, key=lambda x: int(re.search(r"(\d{4})-(\d{2})", x).group(2)))

    if len(files) != 12:
        print(f"[!] Skipping {area_key}  expected 12 temporal images but found {len(files)}")
        continue

    stack = []

    for fname in files:
        path = os.path.join(image_folder, fname)
        img = fm.readGeoTIFFD(path)  # (W, H, bands)
        img = np.transpose(img, (2, 0, 1))
        stack.append(img)

    image_stack = np.stack(stack, axis=0)  # (bands, time, W, H)
    _, _, W, H = image_stack.shape

    # Load corresponding mask
    matching_mask = [m for m in os.listdir(mask_folder)
                     if m.startswith(area_key) and m.endswith('_buffer10.tif')]
    # matching_mask = [m for m in os.listdir(mask_folder)
    #                  if m.startswith(area_key) and m.endswith('_allTouched.tif')]
    if not matching_mask:
        print(f"[!] No mask found for area {area_key}")
        continue

    mask_path = os.path.join(mask_folder, matching_mask[0])
    mask, geotransform, projection = fm.readGeoTIFFD(mask_path, metadata=True)  # (W, H)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = np.where(mask==255, 1, 0) 

    # Extract geotransform parameters
    origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform
    H, W = mask.shape

    #  Clip into 256x256 tiles 
    tile_idx = 0
    for y in range(0, W - clip_size + 1, clip_size):
        for x in range(0, H - clip_size + 1, clip_size):
            patch_img = image_stack[:, :, y:y+clip_size, x:x+clip_size]
            patch_mask = mask[y:y+clip_size, x:x+clip_size]

            # Compute new top-left coordinates
            new_origin_x = origin_x + x * pixel_width
            new_origin_y = origin_y + y * pixel_height

            # Create new geotransform for this tile
            new_geotransform = (
                new_origin_x,
                pixel_width,
                0.0,
                new_origin_y,
                0.0,
                pixel_height
            )

            img_out_path = os.path.join(output_image_folder, f"{area_key}_tile{tile_idx:02d}.tif")
            mask_out_path = os.path.join(output_mask_folder, f"{area_key}_tile{tile_idx:02d}.tif")

            tifffile.imwrite(img_out_path, patch_img.astype(np.float32))
            #fm.writeGeoTIFFD(img_out_path, patch_img.astype(np.uint8), new_geotransform, projection)
            fm.writeGeoTIFF(mask_out_path, patch_mask.astype(np.uint8), new_geotransform, projection)
            #tifffile.imwrite(mask_out_path, patch_mask.astype(np.uint8))

            tile_idx += 1

    print(f"{tile_idx} tiles saved for area: {area_key}")