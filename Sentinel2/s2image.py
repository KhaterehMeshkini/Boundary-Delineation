import os, imageio
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import filemanager as fm
from satimage import SATimg
from scipy.ndimage import binary_dilation as bindilation 

class S2img(SATimg):
    # self._metadata

    #-----------------------------------------------------------------------------------------------#
    #CONSTRUCTOR
    def __init__(self, features=None, temppath=None):
        #INITIALIZE BASIC METADATA
        super().__init__()
        self._metadata['resolution'] = 10

        #CHECK IF FEATUREPATHS HAVE BEEN PASSED
        if features:
            self._populate(features, temppath)
 
    def _populate(self, features, temppath=None):        
        #STORE PATHS
        self._storepaths(features)

        #GET METADATA FROM FN
        self._getmetadata()

        #GET TEMPPATH
        self._gettemppath(temppath)

        if self.flag():
            self._savemetadata()
        else:
            self._loadmetadata()
        
    #-----------------------------------------------------------------------------------------------#
    def _storepaths(self, features):
        for key,path in features.items():
            ftname = self.translate(key)
            self._metadata['featurepath'][ftname] = path

    #-----------------------------------------------------------------------------------------------#
    def _getmetadata(self):
        #SMALL REDUNDANT CHECK
        fd = self.featurepath()
        if (len( fd )>0):
            key = next(iter(fd))
            fp = fd[key]
            self._getinfo(fp)

    def _getinfo(self, filepath):
        #GET FILENAME OF A FEATURE         
        fn = os.path.split(filepath)[1]
        
        #PRAPARE LIST OF USEFULL INFORMATION
        info = fn.split('.')[0]
        info = info.split('_')

        #GET BASIC METADATA
        self._gettile(info)
        self._getdate(info)        

    def _gettile(self, info):
        """There are two slightly different naming conventions, one that begins with "T" (for "Tile")
        and another that simply displays the tile-string. Both conventions need to be checked for."""
        potential = [f for f in info if (len(f)==6)]
        potential = [f for f in potential if f[0]=='T']
        #Check for old naming convention
        if ( len(potential)==0 ):
            potential = [f for f in info if (len(f)==5)]
            potential = [f for f in potential if (f[0:2].isdigit() & f[2:].isalpha())]
            if ( len(potential)==0 ):
                raise Exception('No valid tile-information was found!')
            else:
                tile = 'T' + potential[0]
        else:
            tile = potential[0]

        self._metadata['tile'] = tile

    def _getdate(self, info):
        potential = [f for f in info if len(f)==15] #filters right data format
        potential = [f.split('T') for f in potential] #splits date and time 
        date = min( potential, key=lambda x: x[0] )[0] #get the earliest date(sicen there might be more than one)     
        potential = [f for f in potential if (f[0]==date)] #get all date+time istances of ealiest date       
        self._metadata['date'], self._metadata['time'] = max( potential, key=lambda x: x[1] )
   
    #-----------------------------------------------------------------------------------------------#
    def _gettemppath(self, temppath):
        if (temppath==None):
            features = self.featurepath()
            key = next( iter( features) )
            path = features[key]
            fp = os.path.split(path)[1]
        else:
            fp = fm.joinpath( temppath, self.name() )  
        self._metadata['temppath'] = fm.check_folder(fp)
    
    #-----------------------------------------------------------------------------------------------#
    def _getmask(self):
        """How Sentinel-2 Scene Classification (SCL) is computed: 
        https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm 
                
        MASK-Legend:
        -1:Defective
        -2:NAN        
        -3:Clouds
        -4:CloudShadows
        -5:Snow
        """        
        #READ SCL
        scl = self.feature('SCL',store=False)
        
        #PREPARE MASK
        height,width = scl.shape
        mask = np.zeros( (height,width), dtype=np.uint8 )

        #GET DEFECTIVE-MASK
        layer = (scl==1)
        mask[layer] = 1

        #GET NAN-MASK (Not-A-Number)
        layer = (scl==0)
        mask[layer] = 2

        #GET CLOUD-SHADOW-MASK
        layer = (scl==3)        
        mask[layer] = 4

        #GET CLOUD-MASK: medium-prob, high-prob, cirrus
        layer = (scl==8) | (scl==9) | (scl==10)        
        mask[layer] = 3        

        #GET SNOW-MASK
        layer = (scl==11)
        mask[layer] = 5
        scl = None

        self._metadata['invalidpixnum'] = np.count_nonzero(mask)
        self._metadata['nanpixnum'] = np.count_nonzero(mask==2)
        self._metadata['cloudypixnum'] = np.count_nonzero((mask==3) | (mask==4))
        self._metadata['totpixnum'] = height*width
        #SAVE MASK
        fp = fm.joinpath(self.temppath(), 'MASK.npy')
        np.save(fp, mask)

    def InvalidPixNum(self):
        return self._metadata['invalidpixnum']

    def NANPixNum(self):
        return self._metadata['nanpixnum']
    
    def CloudyPixNum(self):
        return self._metadata['cloudypixnum']

    def TotalPixNum(self):
        return self._metadata['totpixnum']
    #-----------------------------------------------------------------------------------------------#
    #METHODS FOR I/O OPERATIONS
    def copy(self, newpath):
        print(self.name(), ': deep-copy to new path... ', end='\r')
        newtemppath = fm.check_folder(newpath, self.name())

        #COPY VARIABLES
        newimg = S2img()
        newimg._metadata = self._metadata
        newimg._metadata['temppath'] = newtemppath

        #SAVE METADATA: don't copy it!
        newimg._savemetadata()        

        #COPY FEATURES
        fn = [f for f in os.listdir(self.temppath()) if f.endswith('.npy')]
        for feature in fn:
            print(self.name(), ': deep-copy to new path... Copying: ', feature, '         ',end='\r')
            oldpath = fm.joinpath(self.temppath(), feature)
            newpath = fm.joinpath(newimg.temppath(), feature)
            np.save(newpath, np.load(oldpath))
        newimg.flag(flagis=False) #reset flag
        print(self.name(), ': deep-copy to new path... DONE!                  ')

        return newimg

    def updatefeature(self, string, matr):
        #GET PATH TO STORED FEATURE/BAND
        name = self.translate(string)
        fp = fm.joinpath(self.temppath(), name+'.npy')

        #VERIFY THAT FEATURE EXISTS
        if os.path.isfile(fp):     

            #CHECK THAT SHAPE MATCHES   
            ref = self.feature(name)
            if (matr.shape==ref.shape):
                #UPDATE FLAG(Do it first, maybe something goes wrong)                  
                self.flag(flagis=True)                

                #UPDATE FEATURE
                np.save(fp, matr)

            else:
                raise RuntimeError('Cannot update feature: dimensions do not match!')
        else:
                raise RuntimeError('Cannot update feature: no file named "', name,'.npy" was found!')
  
    #-----------------------------------------------------------------------------------------------#
    #USEFULL TOOLS
    def name(self):
        name = self._metadata['tile'] + '_' + self._metadata['date'] + 'T' + self._metadata['time']
        return name

    def nanmask(self):
        img = self.feature('MASK')
        return (img==2)
       
    def translate(self, string):
        #SETUP DICTIONARY
        dictionary = {}
        dictionary['B01'] = ['B1','b1','B01','b01','Coastal Aerosol','Aerosol','aerosol']
        dictionary['B02'] = ['B2','b2','B02','b02','BLUE','blue']
        dictionary['B03'] = ['B3','b3','B03','b03','GREEN','green']
        dictionary['B04'] = ['B4','b4','B04','b04','RED','red']
        dictionary['B05'] = ['B5','b5','B05','b05','RE1']
        dictionary['B06'] = ['B6','b6','B06','b06','RE2']
        dictionary['B07'] = ['B7','b7','B07','b07','RE3']
        dictionary['B08'] = ['B8','b8','B08','b08','NIR','nir']
        dictionary['B8A'] = ['B8A','b8A','B8a','b8a']
        dictionary['B09'] = ['B9','b9','B09','b09','Water Vapor','water vapor','vapor']
        dictionary['B11'] = ['B11','b11','SWIR1','swir1','1600']
        dictionary['B12'] = ['B12','b12','SWIR2','swir2','2200']
        dictionary['NDVI'] = ['NDVI','ndvi']
        dictionary['SCL'] = ['SCL','scl']    
        dictionary['RESI'] = ['RESI','resi']   
        dictionary['NDSI'] = ['NDSI','ndsi']     
        dictionary['MASK'] = ['MASK','mask','Mask']
        dictionary['RGB'] = ['RGB','rgb']

        #SLOW-SEARCH
        for key in list(dictionary.keys()):
            for s in dictionary[key]:
                if s==string:
                    return key
        
        print('SatImage has no band named "',string,'"!')
        return None

    def feature(self, string, **kwargs):
        dtype = kwargs.get('dtype', None)
        name = self.translate(string)
        store = kwargs.get('store', True)

        #MASK-FEATURE IS SPECIAL CASE
        if name=='MASK':
            if self.temppath():
                fp = fm.joinpath(self.temppath(),'MASK.npy')                
                if (os.path.isfile(fp)==False):
                    self._getmask()
                if dtype:
                    matr =  np.load(fp).astype(dtype) #astype raises error if dtype is invalid
                else:
                    matr = np.load(fp)                    
        elif name=='SCL':
            matr = super().feature(name, dtype=dtype, upscale='nearest_neighbor', store=store)
        else:
            matr = super().feature(name, dtype=dtype, store=store)

        return matr

    def feature_resc(self, string, **kwargs):
        dtype = kwargs.get('dtype', None)
        name = self.translate(string)
        store = kwargs.get('store', True)

        #MASK-FEATURE IS SPECIAL CASE
        if name=='MASK':
            if self.temppath():
                fp = fm.joinpath(self.temppath(),'MASK.npy')                
                if (os.path.isfile(fp)==False):
                    self._getmask()
                if dtype:
                    matr =  np.load(fp).astype(dtype) #astype raises error if dtype is invalid
                else:
                    matr = np.load(fp)                    
        elif name=='SCL':
            matr = super().feature(name, dtype=dtype, upscale='nearest_neighbor', store=store)

        elif self._metadata['resolution'] != 10:
            rescale(name, scale, interpolation_type='bilinear')
            matr = super().feature(name, dtype=dtype, store=store)
        
        else:
            matr = super().feature(name, dtype=dtype, store=store)    

        return matr    
