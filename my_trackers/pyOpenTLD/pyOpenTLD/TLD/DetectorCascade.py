from math import floor
from multiprocessing import Pool
import time
TLD_WINDOW_SIZE = 5;
TLD_WINDOW_OFFSET_SIZE = 6

def sub2idx(x,y,widthstep):
    return (int(floor((x)+0.5) + floor((y)+0.5)*(widthstep)))

class DetectorCascade:
    numScales = 0
    scales = []
    minScale = -10
    maxScale = 10
    useShift = 1
    shift = 0.1
    minSize = 25
    numFeatures = 10
    numTrees = 13
    imgWidth = -1
    imgHeight = -1
    imgWidthStep = -1
    objWidth = 1
    objHeight = 1
    numWindows = 0
    windows = []
    windowOffsets = []
    initialised = False
    
    from Clustering import Clustering
    from DetectionResult import DetectionResult
    from EnsembleClassifier import EnsembleClassifier
    from ForegroundDetector import ForegroundDetector
    from nNClassifier import NNClassifier, NormalizedPatch
    from VarianceFilter import VarianceFilter
    
    foregroundDetector = ForegroundDetector()
    varianceFilter = VarianceFilter()
    ensembleClassifier = EnsembleClassifier()
    clustering = Clustering()
    nnClassifier = NNClassifier()
    detectionResult = DetectionResult()
    
    def __init__(self):
        pass
        
    def init(self):
        print "DetectorCascade init"
        self.initWindowsAndScales()
        self.initWindowOffsets()
        self.propagateMembers()
        self.ensembleClassifier.init()
        self.initialised = True
        
    def propagateMembers(self):
        print "propogateMembers"
        self.detectionResult.init(self.numWindows,self.numTrees)
        self.varianceFilter.windowOffsets = self.windowOffsets
        self.ensembleClassifier.windowOffsets = self.windowOffsets
        self.ensembleClassifier.imgWidthStep = self.imgWidthStep
        self.ensembleClassifier.numScales = self.numScales
        self.ensembleClassifier.scales = self.scales
        self.ensembleClassifier.numFeatures = self.numFeatures
        self.ensembleClassifier.numTrees = self.numTrees
        self.nnClassifier.windows = self.windows
        self.clustering.windows = self.windows
        self.clustering.numWindows = self.numWindows

        self.foregroundDetector.minBlobSize = self.minSize*self.minSize

        self.foregroundDetector.detectionResult = self.detectionResult
        self.varianceFilter.detectionResult = self.detectionResult
        self.ensembleClassifier.detectionResult = self.detectionResult
        self.nnClassifier.detectionResult = self.detectionResult
        self.clustering.detectionResult = self.detectionResult
        
        print id(self.foregroundDetector.detectionResult)
        print id(self.detectionResult)
        print id(self.varianceFilter.detectionResult)
        print id(self.ensembleClassifier.detectionResult)
        print id(self.nnClassifier.detectionResult)
        print id(self.clustering.detectionResult)
        
        #raw_input("propogate members:")
    
    def initWindowsAndScales(self):
        scanAreaX = 1
        scanAreaY = 1
        
        scanAreaW = self.imgWidth-1
        scanAreaH = self.imgHeight-1
        scaleIndex = 0
        windowIndex = 0
        self.scales = [[0,0]]*(self.maxScale-self.minScale+1)
        self.numWindows = 0
        
        for i in xrange(self.minScale,self.maxScale+1):
            scale = pow(1.2,i)
            w = int(self.objWidth*scale)
            h = int(self.objHeight*scale)
            if self.useShift:
                ssw = max(1,w*self.shift)
                ssh = max(1,h*self.shift)
            else:
                ssw = 1
                ssh = 1
            if w < self.minSize or h < self.minSize or w > scanAreaW or h > scanAreaH: 
                continue
            self.scales[scaleIndex][0] = w
            self.scales[scaleIndex][1] = h
            scaleIndex+=1
            self.numWindows += floor(float(scanAreaW - w + ssw)/ssw)*floor(float(scanAreaH - h + ssh) / ssh)
            self.numWindows = int(self.numWindows)
        print "self.numWindows",
        print self.numWindows
        print "scaleIndex",
        print scaleIndex
        self.numScales = scaleIndex
        self.windows = [0]*(TLD_WINDOW_SIZE*self.numWindows)
        print "list made"
        print self.numScales,
        print "self.numScales"
        for scaleIndex in xrange(self.numScales):
            w = self.scales[scaleIndex][0]
            h = self.scales[scaleIndex][1]
            if self.useShift:
                ssw = max(1,w*self.shift)
                ssh = max(1,h*self.shift)
            else:
                ssw = 1
                ssh = 1
            y = scanAreaY
            while y + h <= scanAreaY +scanAreaH:
                x = scanAreaX
                while x + w <= scanAreaX + scanAreaW:
                    index = TLD_WINDOW_SIZE*windowIndex
                    self.windows[index:index+5] = x, y, w, h
                    windowIndex+=1
                    x+=ssw
                y+=ssh
        print "initWindowandScales end"
        #assert(windowIndex == self.numWindows)
        
    def initWindowOffsets(self):
        print "initWindowOffsets start"
        self.windowOffsets = [0]*TLD_WINDOW_OFFSET_SIZE*self.numWindows
        off = []
        
        windowSize = TLD_WINDOW_SIZE
        print self.numWindows,
        print "self.numWindows"
        t1 = time.time()
        for i in xrange(self.numWindows):
            index = windowSize*i
            window = self.windows[index:index+5]
            if len(window) < 5:
                continue
            off.append(sub2idx(window[0]-1,window[1]-1,self.imgWidthStep))
            off.append(sub2idx(window[0]-1,window[1]+window[3]-1,self.imgWidthStep))
            off.append(sub2idx(window[0]+window[2]-1,window[1]-1,self.imgWidthStep))
            off.append(sub2idx(window[0]+window[2]-1,window[1]+window[3]-1,self.imgWidthStep))
            off.append(window[4]*2*self.numFeatures*self.numTrees)
            off.append(window[2]*window[3])
        self.windowOffsets[:len(off)]=off
        t2 = time.time()
        #print i
        #print t2-t1,
        #print "time taken"
        
    def detect(self, img):
        print "DetctorCascade detect"
        '''
        print id(self.foregroundDetector.detectionResult.posteriors)
        print id(self.detectionResult.posteriors)
        print id(self.varianceFilter.detectionResult.posteriors)
        print id(self.ensembleClassifier.detectionResult.posteriors)
        print id(self.nnClassifier.detectionResult.posteriors)
        print id(self.clustering.detectionResult.posteriors)
        '''
        from TLDUtil import tldIsInside
        self.detectionResult.reset()
        if not self.initialised:
            print "self.initialised false"
            return
        self.foregroundDetector.nextIteration(img)
        self.varianceFilter.nextIteration(img)
        self.ensembleClassifier.nextIteration(img)
        
        #multiprocessing stuff .. what ??
        print "detecting"
        print self.numWindows
        for i in xrange(self.numWindows):
            self.ensembleClassifier.detectionResult = self.detectionResult
            self.clustering.detectionResult = self.detectionResult
            self.varianceFilter.detectionResult = self.detectionResult
            index = TLD_WINDOW_SIZE*i
            window = self.windows[index:index+4]
            if len(window) < 4:
                break
            if self.foregroundDetector.isActive():
                #print "foregroundDetector Active"
                isInside = False
                #print "inInside False"
                for j in xrange(len(self.detectionResult.fgList)):
                    print self.detectionResult.fgList[j:j+4],
                    print fgList
                    bgBox = self.detectionResult.fgList[j:j+4]
                    if tldIsInside(window, bgBox):
                        isInside = True
                    else:
                        isInside = False
                if not isInside:
                    self.detectionResult.posteriors[i] = 0
                    continue
            if not self.varianceFilter.filter(i):
                #print "not self.varianceFilter"
                self.detectionResult.posteriors[i] = 0
                continue
            if not self.ensembleClassifier.filter(i):
                #print "not self.ensembleClassifier"
                continue
            if not self.nnClassifier.filter(img,i):
                #print "not self.nnClassifier"
                continue
            
            self.detectionResult.confidentIndices.append(i)
        self.clustering.detectionResult = self.detectionResult
        #print "self.clustering.clusterConfidentIndices()?"
        self.clustering.clusterConfidentIndices()
        self.detectionResult = self.clustering.detectionResult
        self.detectionResult.containsValidData = True
        '''
        print id(self.foregroundDetector.detectionResult)
        print id(self.detectionResult)
        print id(self.varianceFilter.detectionResult)
        print id(self.ensembleClassifier.detectionResult)
        print id(self.nnClassifier.detectionResult)
        print id(self.clustering.detectionResult)
        '''
    def cleanPreviousData(self):
        self.detectionResult.reset()
        
    def release(self):
        if not self.initialised:
            return
        self.initialised = False
        self.foregroundDetector.release()
        self.ensembleClassifier.release()
        self.nnClassifier.release()
        self.clustering.release()
        self.numWindows = 0
        self.numScales = 0
        self.scales = []
        self.windows = []
        self.windowOffsets = []
        self.objWidth = 1
        self.objHeight = 1
        self.detectionResult.release()
