import time

import numpy as np

import filemanager as fm

import os
import tifffile as tiff

#---------------------------------------------------------------------------------------------------#
def manager(tile, **kwargs):
    #SETUP VARIABLES
    info = kwargs.get('info', True)
    year = kwargs.get('year', None)
    Imagepath = fm.check_folder(kwargs.get('savepath', None), 'Images')
    Maskpath = fm.check_folder(kwargs.get('savepath', None), 'Masks')
    RGBpath = fm.check_folder(kwargs.get('savepath', None), 'RGB')
    Timepath = fm.check_folder(kwargs.get('savepath', None), 'Time')
    Schiantipath = kwargs.get('Schiantipath', None)


    #GET FEATURES
    yearts,_,_ = tile.gettimeseries(year=year, option='default')

    if len(yearts) != 0:
        _feature(yearts, Imagepath, Maskpath, RGBpath, Timepath, **kwargs)
     

#---------------------------------------------------------------------------------------------------#
#COMPUTE INDEX  
def _feature(ts, path1, path2, path3, path4, **kwargs):

    info = kwargs.get('info',True)
    ts_length = kwargs.get("ts_legth", len(ts) )
    Schiantipath = kwargs.get('Schiantipath', None)
    

    if info:
        print('Extracting features for each image:')
        t_start = time.time()

    #Get some information from data
    height, width = ts[0].feature('B04').shape
    geotransform, projection = fm.getGeoTIFFmeta( ts[0].featurepath()['B04'] )
    

    ts = sorted(ts, key=lambda x: x.InvalidPixNum())[0:ts_length]
    totimg = len(ts)
    totbands = 8
    totpix = height * width
    images = []

    schianti_mask = fm.readGeoTIFF(Schiantipath)
    # Replace values less than 3 (danno alta) with zero
    schianti_mask[schianti_mask < 3] = 0

    
    
    #Compute Index Statistics   
    for idx,img in enumerate(ts):
        if info:        
            print('.. %i/%i      ' % ( (idx+1), totimg ), end='\r' )   

        cloudp_perc = img.CloudyPixNum()/totpix 
        cloudp_perc *= 100
        
        

        if cloudp_perc <= 2: 
              
        
            feature = np.empty((totbands, height, width), dtype=np.uint16)
            #Compute Index
            b1 = img.feature_resc('BLUE', type=np.uint16)
            
    
            b2 = img.feature_resc('GREEN', dtype=np.uint16)
           

            b3 = img.feature_resc('RED', dtype=np.uint16)
            

            b4 = img.feature_resc('RE1', dtype=np.uint16)
            

            b5 = img.feature_resc('RE2', dtype=np.uint16)
            

            b6 = img.feature_resc('NIR', dtype=np.uint16)
            

            b7 = img.feature_resc('SWIR1', dtype=np.uint16)
           

            b8 = img.feature_resc('SWIR2', dtype=np.uint16)
            



            feature[0, ...] = b1
            feature[1, ...] = b2
            feature[2, ...] = b3
            feature[3, ...] = b4
            feature[4, ...] = b5
            feature[5, ...] = b6
            feature[6, ...] = b7
            feature[7, ...] = b8

            images.append(feature)

    ts_images = np.stack(images, axis = 0)
    print(ts_images.shape)

    a, _,_,_ =np.shape(ts_images)
        

    tile_size = 128


    
    # Sliding window for clipping
    tile_count = 0
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            # Define tile bounds
            tile_mask = schianti_mask[i:i+tile_size, j:j+tile_size]
            
            # Skip tiles where the mask is entirely NaN
            if tile_mask.shape != (tile_size, tile_size) or np.all(tile_mask == 0):
                continue
            

            tile_array = ts_images[:, :, i:i+tile_size, j:j+tile_size]
            tile_array_rgb = ts_images[0, 0:3, i:i+tile_size, j:j+tile_size]
            tile_array_rgb_t = np.transpose(tile_array_rgb, (1,2,0)) 

                
            # Save the tiles
            array_tile_path = os.path.join(path1, f"tile_{tile_count}.tif")
            mask_tile_path = os.path.join(path2, f"tile_{tile_count}.tif")
            RGB_tile_path = os.path.join(path3, f"tile_{tile_count}.tif")

            #Save features
            # Calculate new origin
            new_origin_x = geotransform[0] + j * geotransform[1]
            new_origin_y = geotransform[3] + i * geotransform[5]

            # Create the new geotransform
            new_geotransform = [
                new_origin_x,
                geotransform[1],
                geotransform[2],
                new_origin_y,
                geotransform[4],
                geotransform[5],
            ]
            #print(new_geotransform)

            if tile_count == 89:
                for n in range(a):

                    tile_array_time = ts_images[n, :, i:i+tile_size, j:j+tile_size]
                    tile_array_time_t = np.transpose(tile_array_time, (1,2,0)) 
                    Time_tile_path = os.path.join(path4, f"tile_Time_{n}.tif")

                    fm.writeGeoTIFFD(Time_tile_path, tile_array_time_t, new_geotransform, projection)



            #tiff.imsave(array_tile_path, tile_array, dtype=np.uint16)  # Adjust dtype as needed                
            #fm.writeGeoTIFFD(RGB_tile_path, tile_array_rgb_t, new_geotransform, projection)
            #fm.writeGeoTIFF(mask_tile_path, tile_mask, new_geotransform, projection)
            
            tile_count += 1


    if info:
        t_end = time.time()
        print('\nMODULE 1: clipping images..Took ', (t_end-t_start)/60, 'min')
    
