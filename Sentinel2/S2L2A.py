import os, imageio, zipfile, time
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import filemanager as fm
from Sentinel2.s2image import S2img

##################################################################################################
# Sentinel-2 L2A Image 
class S2L2Aimg(S2img):


    def _getinfo(self, filepath):
        #GET FILENAME OF A FEATURE    
        info = filepath.split('.')
        info = [f.split(os.sep) for f in info]
        info = [inner for outer in info for inner in outer]
        info = [f.split('_') for f in info]
        info = [inner for outer in info for inner in outer if len(inner)>0]

        #GET BASIC METADATA
        self._gettile(info)
        self._getdate(info)  

    #-----------------------------------------------------------------------------------------------#
    #USEFULL TOOLS
    def readL2A(self, l2apath, temppath):
        features = {}

        frmt = l2apath.split('.')[-1]
        #OPEN ZIP-FILE
        if (frmt=='zip'):
            try:
                zipf = zipfile.ZipFile( l2apath, 'r' )
            except:
                raise IOError("Unable to open ZIP-file!")
            flist = zipf.namelist()
            zipf.close()  

            fnames = [f for f in flist if (f.endswith('.jp2') | f.endswith('.tif'))]   
        #NAVIGATE SAFE-FOLDER
        elif (frmt=='SAFE'):
            fnames = []
            for _, _, filenames in os.walk(l2apath):
                for f in filenames:
                    if (f.endswith('.jp2') | f.endswith('.tif')):                        
                        fnames.append(f)

        bandnames = {}
        bandnames['10m'] = ['B02', 'B03', 'B04', 'B08']
        bandnames['20m'] = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL']
        #bandnames['60m'] = ['B01', 'B09']

        #READ (AND PROCESS) EACH BAND
        for resolution in list( bandnames.keys() ):
            res_bandnames = [f for f in fnames if resolution in f] #fn with correct resolution

            for band in bandnames[resolution]:
                #GET RIGHT BAND-NAME
                fn = [f for f in res_bandnames if band in f]
                if len(fn)==1:
                    fn = fn[0]
                else:
                    raise IOError('Unable to find band "%s" in %s!!' %( band, self.name() ) )

                #READ BAND-FILE

                if (frmt==('zip')):                    
                    fp = "/vsizip/%s/%s" % (l2apath, fn)
                else:
                    fp = "%s/%s" % (l2apath, fn)
                features[band] = fp
        
        self._populate(features,temppath)
        if self.flag(): 
            self._getmask()
            self._savemetadata() 
            self.flag(False)      
        
        return self

##################################################################################################
# Sentinel-2 L2A Time Series 
class L2Ats:
    """
    FEATURES:
     _ts: it's a list of S2IMG-instances; the following methods are implemented to manage this feature:
        __getitem__: allows to directly index the S2TS to return the corresponding S2IMG in the list;
        __len__: returns the length of 
    """     
    #self._ts
    #self._metadata
    #--------------------------------------------------------------------------------------------#
    #OVERLOADED OPERATOR(S)
    def __init__(self, temppath=None, filepaths=None):
        self._metadata = {}
        if temppath:
            self._metadata['temppath'] = temppath        
            self._ts = []
             
            #POPULATE TS  
            totimg = len(filepaths) 
            done = {}   
            duplicates = {}                         
            for idx,fp in enumerate(filepaths):               
                print('Reading image %i/%i   ' %((idx+1), totimg) , end='\r') 
                #READ IMAGE                 
                img = S2L2Aimg().readL2A(fp, temppath)

                #CHECK FOR DUPLICATES
                key = img.name()
                if key in done.keys():
                    duplicates[key] = [done[key], fp]
                else:
                    done[img.name()] = fp
                
                #APPEND IMG TO TS           
                self._ts.append( img )    
            print('\nReading image: DONE!')                          
                            
    def _matchfeatures(self, features):
        temp = {}
        temp.update(features)        
        
        #GET MISSING FILES
        feature_list = list(temp.keys())
        ref = temp[feature_list[0]]
        missing = []
        for idx in range(1, len(feature_list)):  
            comp = temp[feature_list[idx]]
            missing += list(set(ref.keys()) - set(comp.keys()) )
            missing += list(set(comp.keys()) - set(ref.keys()) )

        #REMOVE MISSING FILES
        if ( len(missing)>0 ):
            for feature in feature_list:
                for key in missing:
                    temp[feature].pop(key, None)
            print('The following dates are missing:')
            print(missing)

        scl = temp.pop('SCL', None)
        return temp, scl

    #-----------------------------------------------------------#
    #LIST SPECIFIC METHODS
    def __len__(self):
        return len(self._ts)

    def __getitem__(self, key):
        return self._ts[key]
        
    def append(self, img):
        if type(img) is S2L2Aimg:            
            self._ts.append(img)
        else:
            raise Exception('Type Error: non-s2img type was passed!')

    #--------------------------------------------------------------------------------------------#
    #LIST SORTING METHODS
    def sort(self, ts=None):
        if (ts==None):
            ts = self._ts
        ts.sort(key= self._sortKey)        

    def sorted(self, reference):
        sl = sorted(self._list, key=lambda x:self.euclideandate(reference, x))
        return sl

    def _sortKey(self, e):
        return e.date(ordinal=True)

    def euclideandate(self, ref, img):
        refdate = ref.date(ordinal=True)
        imgdate = img.date(ordinal=True)
        return abs(refdate-imgdate)

    #--------------------------------------------------------------------------------------------#
    #METHODS RETURNING CLASS INFORMATION    
    def tile(self):
        return self[0]._metadata['tile']

    def temppath(self):
        return self._metadata['temppath']
    #--------------------------------------------------------------------------------------------#
    def getdays(self, firstday=None):        
        ts = self._ts        
        self.sort(ts)
        if firstday:
            firstday = fm.string2ordinal(firstday) - 1
        else:
            firstday = ( ts[0].date() ).toordinal() - 1
        days = [(f.date().toordinal() - firstday) for f in ts]

        return np.array(days)

    def find(self, **kwargs):
        options = {}.fromkeys(['year', 'month','day','month','hour','minute','second'])
        options.update( kwargs )

        results = self._ts
        if options['year']: 
            results = [f for f in results if (f.date().year==options['year'])]
        if options['month']: 
            results = [f for f in results if (f.date().month==options['month'])]
        if options['day']: 
            results = [f for f in results if (f.date().day==options['day'])]
        if options['hour']: 
            results = [f for f in results if (f.date().hour==options['hour'])]
        if options['minute']: 
            results = [f for f in results if (f.date().minute==options['minute'])]
        if options['second']: 
            results = [f for f in results if (f.date().second==options['second'])]
        
        return results

    def getyear(self, year, option='default', buffer=None, fmt="%Y%m%d"):
        if (option=='default'):
            start = fm.string2ordinal(str(year) + '0101', fmt)
            end = fm.string2ordinal(str(year) + '1231', fmt)
        if (option=='farming'):
            y = str( int(year)-1 )
            start = fm.string2ordinal(str(y) + '1111', fmt)
            end = fm.string2ordinal(str(year) + '1110', fmt)
        if buffer:
            start -= buffer
            end += buffer

        #CREATE EMPTY TS OBJECT
        TS = L2Ats()
        #COPY METADATA
        TS._metadata.update(self._metadata)  
        TS._ts = [f for f in self._ts if ((f.date(ordinal=True)>=start) & (f.date(ordinal=True)<=end))]
        
        return TS, fm.ordinal2string(start), fm.ordinal2string(end)
    #--------------------------------------------------------------------------------------------#
    #METHODS FOR DATA MANIPULATION
    def cropdataset(self, x, y, savepath=None):
        """
        In order to properly crop images belonging to the same TS, follow this example: 
            sp = fm.check_folder(datapath,"CROPPED")
            y = (6000,7000)
            x = (7000,8000)
            tile.gettimeseries().cropdataset(y=y, x=x, savepath=sp)
        """
        coordinates = (x[0],x[1], y[0],y[1])
        #ROOT SAVEPATH
        if (savepath==None):
            savepath = os.path.split(self.temppath())[0]
        name = '%s_cropped_y%i_%i_x%i_%i' %(self.tile(), y[0], y[1], x[0], x[1])
        savepath = fm.check_folder(savepath, name)        

        #CROP IMAGE-FEATURES
        totimg = len(self)
        resolution = self[0].resolution()
        for idx,img in enumerate(self):
            print('Cropping %i/%i     '%((idx+1),totimg), end='\r') 
            name = img.name()+'.SAFE'
            sp = fm.check_folder(savepath, 'L2A', name)
            ftdict = img.featurepath()
            for _,path in ftdict.items():
                fm.cropGeoTIFF(coordinates,path,sp, resolution=resolution)
     
    #-----------------------------------------------------------------------------------------------#
    #SAVE IMAGE
    def animatedgif(self, feature, savepath, name, frametime=0.2, ts=None):
        #DEFINE TS
        if (ts==None):
            ts = self._ts

        #PREPARE IMAGE SETTINGS
        fp = fm.joinpath(savepath, name+'.gif')
        cmap = plt.get_cmap('jet')

        #FILL ANIMATED-GIF
        with imageio.get_writer(fp, mode='I', duration=frametime) as writer:
            for _, s2img in enumerate(ts):
                img = s2img.feature(feature)
                cmapimg = cmap(img)*255
                writer.append_date(cmapimg)

    def PlotNANandClOUDY(self, **kwargs):
        step = kwargs.get('step',30) 
        year = kwargs.get('year',None)
        self.sort()
        if year:
            ts,start,end = self.getyear(year)
            start = fm.string2ordinal(start)
            end = fm.string2ordinal(end)
        else:            
            ts = self._ts
            start = ts[0].date(ordinal=True)
            end = ts[-1].date(ordinal=True)

        ordinal = list(range(start, end+1, step))
        lbs = [fm.ordinal2string(f, fmt="%Y/%m/%d") for f in ordinal]
        xdates = list( np.array(ordinal) - start )

        cloudp_perc = np.zeros( (len(ts)), dtype=np.float32 )
        nanp_perc = np.zeros( (len(ts)), dtype=np.float32 )
        invalid_perc = np.zeros( (len(ts)), dtype=np.float32 )
        x = np.zeros( (len(ts)), dtype=np.float32 )
        xstart = start
        for idx,img in enumerate(ts):
            totpix = img.TotalPixNum()
            x[idx] = img.date(ordinal=True) - xstart 
            cloudp_perc[idx] = img.CloudyPixNum()/totpix
            nanp_perc[idx] = img.NANPixNum()/totpix
            invalid_perc[idx] = img.InvalidPixNum()/totpix
        cloudp_perc *= 100
        nanp_perc *= 100
        invalid_perc *= 100

        dates = np.array( [f.date(ordinal=True) for f in ts] )
        unique_dates, unique_counts = np.unique(dates, return_counts=True )
        
        #PLOT OF CLOUDY PIXELS
        plt.stem(x,cloudp_perc, use_line_collection=True)
        plt.ylabel("Cloudy Pixel Percentage(%)", fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(xdates, lbs, rotation=40, fontsize=15)       
        plt.show()
        
        #PLOT OF NANs and Unique date count
        color = 'tab:red'
        _,ax1 = plt.subplots()
        ax1.set_ylabel("NAN Pixel Percentage(%)", color=color, fontsize=15)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(xdates)
        ax1.set_xticklabels(lbs)
        ax1.tick_params(axis="x", labelrotation=40)
        ax1.tick_params(axis="both", labelsize=15)
        ax1.stem(x,nanp_perc,
                linefmt='r', markerfmt='ro', use_line_collection=True)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Overlapping Date Count', color=color, fontsize=15)  # we already handled the x-label with ax1
        ax2.stem((unique_dates-start), unique_counts, 
                linefmt='c', markerfmt='co', use_line_collection=True)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=15)
        plt.show()       
               
        #PLOT INVALID PIXEL PERCENTAGE
        plt.stem(x,invalid_perc, use_line_collection=True)
        plt.ylabel("Overall Invalid Pixel Percentage(%)", fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(xdates, lbs, rotation=40, fontsize=15)
        plt.show()

        pass



##################################################################################################
# Sentinel-2 L2A Tile Time Series 
class L2Atile:
    """
    This object populates and manages the entire TS for a given tile.
    Indexing allows to return the appropriate S2TS
    """
    
    def __init__(self, temppath, filepaths):
        #SETUP BASIC METADATA
        self._metadataconstructor(temppath, filepaths)         

        #INITILIZE S2TS
        self._metadata['ts'] = L2Ats(self.temppath(), filepaths)
        self._metadata['ts'].sort()
     
    def _metadataconstructor(self, temppath, filepaths):
        self._metadata = {
                        'tile': 'None',
                        'temppath': None,
                        'ts': L2Ats(),
                        'duplicates': None
                        }
        #GET TILE
        if (len(filepaths)>0):
            fp = filepaths[0]
            filename = os.path.split(fp)[1]
            self._metadata['tile'] = _gettile(filename) 
        
        #GET TEMPPATH
        self._metadata['temppath'] = fm.check_folder(temppath, 'numpy', self._metadata['tile'])

        #-----------------------------------------------------------#
    #---------------------------------------------------------------------------------------------------#
    #LIST SPECIFIC METHODS
    def __len__(self):
        return len(self.gettimeseries())

    def __getitem__(self, key):
        return self.gettimeseries()[key]          

    #---------------------------------------------------------------------------------------------------#
    def temppath(self):
        return self._metadata['temppath']

    def tile(self, frmt=None):
        t = self._metadata['tile']
        if (frmt==None):            
            return t
        elif (frmt=='short'):
            return t[1:]
        else:
            raise IOError('Passed format option "%s" is not supported!' %(frmt))
     
    def gettimeseries(self, **kwargs):        
        year = kwargs.get('year',None)
        option = kwargs.get('option','default')
        buffer = kwargs.get('buffer',None)
        fmt = kwargs.get('fmt',"%Y%m%d")        

        ts = self._metadata['ts']
        if year:
            ts, start, end = ts.getyear(year, option, buffer, fmt )
            return ts, start, end
        else:
            return ts

#---------------------------------------------------------------------------------------------------#
def getTileList(datapath):
    #GET ALL .ZIP/.SAFE FILEPATHS
    filepaths = []
    for rootname, _, filenames in os.walk(datapath):
        for f in filenames:
            if (f.endswith('.zip')):
                fp = fm.joinpath(rootname, f)
                filepaths.append(fp)
    if (len(filepaths)==0): 
        for rootname, _, _ in os.walk(datapath):        
            if (rootname.endswith('.SAFE')):                
                filepaths.append(rootname)


    #SORT ALL FILEPATHS IN THE RESPECTIVE TILES
    tiledict = {}
    for fp in filepaths:
        filename = os.path.split(fp)[1]
        tile = _gettile(filename)
        if tile not in tiledict.keys():
            tiledict[tile] = []
        tiledict[tile].append(fp)

    for tile in tiledict.keys():
        print("Tile-%s has been added!" %(tile) ) 

    return tiledict

def _gettile(filename):
    info = filename.split('.')[0]
    info = info.split('_')
    potential = [f for f in info if f[0]=='T']

    #Check for old naming convention
    if ( len(potential)==0 ):
        potential = [f for f in info if (f[0:2].isdigit() & f[2:].isalpha())]
        if ( len(potential)==0 ):
            raise Exception('No valid tile-information was found!')
        else:
            tile = 'T' + potential[0]
    else:
        tile = potential[0]

    return tile

def getdate(filename):
    info = os.path.split(filename)[1]
    info = info.split('.')[0] #remove extension
    info = info.split('_') #split filename
    potential = [f for f in info if len(f)==15] #filters right data format
    potential = [f.split('T') for f in potential] #splits date and time  
    date = min( potential, key=lambda x: x[0] )[0] #get the earliest date(sicen there might be more than one)     
    potential = [f for f in potential if (f[0]==date)] #get all date+time istances of ealiest date
    date, _ = max( potential, key=lambda x: x[1] ) #get the latest time for that date

    return date