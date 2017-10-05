TLD_PATCH_SIZE = 15
TLD_WINDOW_SIZE = 15
class NNClassifier:
    enabled = False
    windows = []
    thetaFP = 0.0
    thetaTP = 0.0
    from DetectionResult import DetectionResult
    detectionResult = DetectionResult()
    falsePositives = []
    truePositives = []
    
    def __init__(self):
        self.thetaFP = 0.5
        self.thetaTP = 0.65
        
    def ncc(self, f1, f2):
        corr = 0
        norm1 = 0
        norm2 = 0
        
        size = TLD_PATCH_SIZE*TLD_PATCH_SIZE
        
        for i in xrange(size):
            corr += f1[i]*f2[i]
            norm1 += f1[i]*f1[i]
            norm2 += f2[i]*f2[i]
            
        return (corr / (norm1*norm2)**0.5 + 1) / 2.0
        
    def classifyPatch(self, patch):
        if not self.truePositives:
            return 0
        if not self.falsePositives:
            return 0
        
        ccorr_max_p = 0
        for i in xrange(len(self.truePositives)):
            ccorr = self.ncc(self.truePositives[i].values, patch.values)
            if ccorr > ccorr_max_p:
                ccorr_max_p = ccorr
        
        ccorr_max_n = 0
        for i in xrange(len(self.falsePositives)):
            ccorr = self.ncc(self.falsePositives[i].values, patch.values)
            if ccorr > ccorr_max_n:
                ccorr_max_n = ccorr
        
        dN = 1-ccorr_max_n
        dP = 1-ccorr_max_p
        distance = float(dN)/(dN+dP)
        return distance
    
    def classifyBB(self, img, bb):
        from TLDUtil import tldExtractNormalizedPatchRect
        patch = NormalizedPatch()
        patch.values = tldExtractNormalizedPatchRect(img, bb)
        distance = self.classifyPatch(patch)
        return distance
    
    def classifyWindow(self, img, windowIdx):
        from TLDUtil import tldExtractNormalizedPatchBB
        patch = NormalizedPatch()
        bbox = self.windows[TLD_WINDOW_SIZE*windowIdx:]
        patch.values = tldExtractNormalizedPatchBB(img, bbox)
        distance = self.classifyPatch(patch)
        return distance
        
    def filter(self, img, windowIdx):
        if not self.enabled:
            return True
        conf = self.classifyWindow(img, windowIdx)
        if conf < self.thetaTP:
            return False
        return True
        
    def learn(self, patches):
        for i in xrange(len(patches)):
            patch = patches[i]
            conf = self.classifyPatch(patch)
            
            if patch.positive and conf < self.thetaTP:
                self.truePositives.append(patch)
            if not patch.positive and conf >= self.thetaFP:
                self.falsePositives.append(patch)
                
    def release(self):
        self.truePositives = []
        self.falsePositives = []
      
class NormalizedPatch:
    values = []
    positive = False
    
    def __init__(self):
        self.values = [0.0]*(TLD_PATCH_SIZE*TLD_PATCH_SIZE)
        self.positive = False
