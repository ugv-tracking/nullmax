from SimpleCV import Image, Camera, Display, VirtualCamera
from pyOpenTLD import *
import sys
W = int(sys.argv[1])
H = int(sys.argv[2])

class PyOpenTLD:
    tld = TLD()
    display = Display()
    cam = VirtualCamera("/home/judy/trackers/pyOpenTLD/video/inputcar.avi","video")
    # cam=Camera()
    threshold = 0
    initialBB = []
    
    def __init__(self):
        self.threshold = 0.5
        self.initialBB = []
        
    def start_tld(self, bb=None):
        img = self.cam.getImage().scale(W,H)
        grey = img.toGray().getBitmap()
        
        self.tld.detectorCascade.imgWidth = grey.width
        self.tld.detectorCascade.imgHeight = grey.height
        self.tld.detectorCascade.imgWidthStep = grey.width*grey.nChannels
        if not bb:
            bb = getBBFromUser(self.cam, self.display)
        print bb
        grey = img.toGray()
        img.drawRectangle(bb[0],bb[1],bb[2],bb[3],width=5)
        img.show()
        self.tld.selectObject(grey, bb)
        skipProcessingOnce = True
        reuseFrameOnce = True
        self.process_open()
        
    def process_open(self):
        img = self.cam.getImage().scale(W,H)
        self.tld.processImage(img)
        if self.tld.currBB:
            print self.tld.currBB
            x,y,w,h = self.tld.currBB
            img.drawRectangle(x,y,w,h,width=5)
        img.show()
        self.process_open()

def getBBFromUser(cam, d):
    p1 = None
    p2 = None
    img = cam.getImage()
    while d.isNotDone():
        try:
            img = cam.getImage().scale(W,H)
            a=img.save(d)
            dwn = d.leftButtonDownPosition()
            up = d.leftButtonUpPosition()
            
            if dwn:
                p1 = dwn
            if up:
                p2 = up
                break

            time.sleep(0.1)
        except KeyboardInterrupt:
            break
    if not p1 or not p2:
        return None
    
    bb = getBB(p1,p2)
    print p1,p2
    print bb
    rect = getRectFromBB(bb)
    return rect

p=PyOpenTLD()
p.start_tld()
