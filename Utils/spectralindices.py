import filemanager as fm 
#from satimage import SATimg 
from skimage.exposure import adjust_log
import numpy as np

#--------------------------------------------------------#
def compute_index(img, string):    
    #CALL TRANSLATION FUNCTION FOR STRING    
    name = translate(string)

    dictionary = index_dictionary()
    if name in dictionary.keys():
        return dictionary[name](img)

def translate(string):
    dictionary = {}
    dictionary['RGB'] = ['RGB','rgb']
    dictionary['NDVI'] = ['NDVI','ndvi']
    dictionary['RESI'] = ['RESI','resi']   
    dictionary['NDSI'] = ['NDSI','ndsi']  
    dictionary['CAI_MS'] = ['CAI_MS','CAI_MULTISPECTRAL','CAI','cai_ms','cai_multispectral','cai']
    dictionary['NDSI'] = ['NDSI','ndsi']
    dictionary['GNDVI'] = ['GNDVI','gndvi']

    #SLOW-SEARCH
    for key in list(dictionary.keys()):
        for s in dictionary[key]:
            if s==string:
                return key
    
    raise Exception('No valid index named "',string,'" found!')

def index_dictionary():
    dictionary = {}
    dictionary['RGB'] = _rgb
    dictionary['NDVI'] = _ndvi
    dictionary['CAI_MS'] = _cai_multispectral
    dictionary['NDSI'] = _ndsi
    dictionary['RESI'] = _resi
    dictionary['GNDVI'] = _gndvi

    return dictionary

#--------------------------------------------------------#
def _rgb(img, band=('RED','GREEN','BLUE')):
    gain = 3
    scale = 1E4
    
    #READ BANDS
    red = img.feature(band[0], dtype=np.float32)
    green = img.feature(band[1], dtype=np.float32)
    blue = img.feature(band[2], dtype=np.float32)    

    #PROCESS RGB
    RGB = np.stack( ( red, green, blue ), axis=2 ) 
    RGB = RGB/scale #normalize float values to [0;1]
    RGB = adjust_log(RGB,gain) # adjust gamma
    RGB[RGB>1] = 1 # clip saturated values
    RGB[RGB<0] = 0

    return RGB


def _ndvi(img):
    #GET BANDS
    red = img.feature('RED', dtype=np.float32)
    nir = img.feature('NIR', dtype=np.float32)

    #COMPUTE INDEX
    denom = nir + red        
    nom = (nir-red)
    index = nom/denom

    #CLIP OUTPUT VALUES
    index[index>1] = 1
    index[index<-1] = -1
    
    return index

def _cai_multispectral(img):
    #GET BANDS
    swir1 = img.feature('1600', dtype=np.float32)
    swir2 = img.feature('2200', dtype=np.float32)

    denom = swir1
    denom[denom==0] = 1e-8

    index = (swir2)/(denom)
    index[index>1E4] = 1E4 #should never be this high though
    index[index<0] = 0   

    return index

def _ndsi(img):
    swir1 = img.feature('1600', dtype=np.float32)
    green = img.feature('GREEN', dtype=np.float32)

    denom = green + swir1
    denom[denom==0] = 1e-8

    index = (green - swir1)/(denom)
    index[index>1] = 1
    index[index<-1] = -1

    return index

def _gndvi(img):
    green = img.feature('GREEN', dtype=np.float32)
    nir = img.feature('NIR', dtype=np.float32)

    denom = nir + green
    denom[denom==0] = 1e-8

    index = (nir - green)/(denom)
    index[index>1] = 1
    index[index<-1] = -1

    return index

def _resi(img):
    veg_re1 = img.feature('RE1', dtype=np.float32)/1E4
    veg_re2 = img.feature('RE2', dtype=np.float32)/1E4
    veg_re3 = img.feature('RE3', dtype=np.float32)/1E4

    index = 0.5*(veg_re1+veg_re3) + veg_re2 

    return index

def _ndi(b1,b2):

    denom = b1 + b2        
    nom = (b1-b2)
    index = nom/denom    

    return index    
