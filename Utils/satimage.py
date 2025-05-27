import os
import numpy as np
from osgeo import gdal
from datetime import datetime
import filemanager as fm
import spectralindices as si

class SATimg:
    # self._metadata

    #-----------------------------------------------------------------------------------------------#
    #CONSTRUCTOR
    def __init__(self):
        #Initialize EMPTY SAT-Image Obj 
        self._metadata = {
            'featurepath': {}, #contains the paths to the original .tif/.jp2 features
            'temppath': None, #path where read data and metadata is stored
            'resolution': None, #reference image resolution for all features
            'shape': None, #image dimensions
            'geotransform': None, 
            'projection': None,           
            'tile': None,
            'date': None,
            'time': None,
        }
    #-----------------------------------------------------------------------------------------------#
    #CONSTRUCTOR HELPER METHODS
    def _loadmetadata(self):
        path = self.temppath()
        name = 'metadata.pkl'
        loadpath = fm.joinpath(path, name)
        self._metadata = fm.loadvar(loadpath)

    def _savemetadata(self):
        path = self.temppath()
        name = 'metadata.pkl'
        savepath = fm.joinpath(path, name)
        fm.savevar(savepath, self._metadata)

    def _getGeoRefMeta(self):
        fd = self._metadata['featurepath']
        key = next(iter(fd))
        fp = fd[key][0]
        geotransform, projection = fm.getGeoTIFFmeta(fp)

        #ADAPT METADATA TO RESOLUTION
        geotransform = (geotransform[0], 
                        self.resolution(),
                        geotransform[2],
                        geotransform[3],
                        geotransform[4],
                        (-1)*self.resolution()
                        )
        #self._metadata['shape'] = matr.shape
        self._metadata['geotransform'] = geotransform
        self._metadata['projection'] = projection

    def translate(self, string):
        dictionary = {}        

        #SLOW-SEARCH
        for key in list(dictionary.keys()):
            for s in dictionary[key]:
                if s==string:
                    return key
        return None

    #-----------------------------------------------------------------------------------------------#
    #RETRIEVE OBJECT INFO
    def date(self, ordinal=False):
        d = datetime.strptime( self._metadata['date'], '%Y%m%d' ).date()
        if (ordinal==True):
            return d.toordinal()
        else:
            return d
    
    def time(self):
        return datetime.strptime( self._metadata['time'], '%H%M%S').time()

    def tile(self):
        return self._metadata['tile']

    def featurepath(self, key=None):
        fd = self._metadata['featurepath']
        if key:
            name = self.translate(key)
            return fd[name]
        else:
            return fd

    def temppath(self):
        return self._metadata['temppath']

    def flag(self, flagis=None):
        flagpath = fm.joinpath(self.temppath(), 'flag.npy')
        if (flagis!=None):
            np.save(flagpath, flagis)
        if os.path.isfile(flagpath):
            return np.load( flagpath )
        else:
            return True

    def geotransform(self):
        if (self._metadata['geotransform']==None): 
            self._getGeoRefMeta()  
            self._savemetadata()         
        return self._metadata['geotransform']

    def projection(self):
        if (self._metadata['projection']==None): 
            self._getGeoRefMeta()
            self._savemetadata()            
        return self._metadata['projection']

    def resolution(self):
        return self._metadata['resolution']

    #-----------------------------------------------------------------------------------------------#
    #USEFULL TOOLS
    def feature(self, name, **kwargs):
        dtype = kwargs.get('dtype', None)
        store = kwargs.get('store', True)
        upscale = kwargs.get('upscale', 'bicubic')
        
        if self.temppath():
            fp = fm.joinpath(self.temppath(), name+'.npy')
            #IF THERE ALREADY IS A NUMPY FILE, LOAD IT
            if os.path.isfile(fp):
                if dtype:
                    matr =  np.load(fp).astype(dtype) #astype raises error if dtype is invalid
                else:
                    matr = np.load(fp)

            #ELSE READ/COMPUTE THE FEATURE, STORE AND RETURN IT
            else: 
                #FEATURE CAN BE READ FROM STORED FEATUREPATH
                if name in self.featurepath().keys():  
                    rp = self.featurepath()[name]
                    matr, geotransform, _ = fm.readGeoTIFF(rp, metadata=True)
                    res = geotransform[1]
                    ratio = int(res/self._metadata['resolution'])
                    if (ratio!=1):
                        matr = fm.rescale(matr, ratio, upscale)
                #FEATURE IS A PRODUCT THAT CAN BE COMPUTED WITH AVAILABLE FEATURES
                else:           
                    matr = si.compute_index(self, name)
                
                #STORE DATA
                if (store==True):
                    np.save(fp, matr)

            #RETURN FEATURE
            return matr.astype(dtype)
        else:
            raise IOError('Invalid "temppath": path was not correctly initialized!')
    
    def index(self, name):
        img = self
        return si.compute_index(img, name)

    def rgb(self):
        return self.index('RGB')
