import numpy as np
import cv2
import glob

cv2.namedWindow("frame",1)

#get from webcam
# cap = cv2.VideoCapture(0)
    
#find params for corner detection via Shi-Tomasi
shi_params = dict( maxCorners = 3000,
                       qualityLevel = 0.05, #determines amount of relevant points determined
                       minDistance = 7,
                       blockSize = 7 )

#find params for Lucas-Kanade optical flow
lk_params = dict( winSize  = (12,12), #larger window means better tracking, but slower computation (more cost)
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#random color gen!!! -- color[i].tolist()
#color = np.random.randint(0,255,(1000000,3))

#take first frame
# ret, old_frame = cap.read()
folder = '/home/judy/vot2016/cars/car1/'
fn_imgs = glob.glob(folder + '/*.jpg')
fn_imgs.sort()
ori_imgs = [cv2.imread(f) for f in fn_imgs]
old_frame = ori_imgs[0]

out_video = cv2.VideoWriter(
    filename='out_lk.avi',
    fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
    fps=30,
    frameSize=tuple([640,480]),
    isColor=True)

#grayscale convert it
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)

# while(1):
for i, ori_img in enumerate(ori_imgs[1:]):
    #we need to continuously find good params to track
    p_old = cv2.goodFeaturesToTrack(old_gray, mask = None, **shi_params)

    #read curr frame and grayscale it
    # ret,frame = cap.read()
    frame = ori_img
    fgray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    #init p_new in case we wanted to use flags
    p_new = np.zeros_like(p_old)
    
    #use Lucas-Kanade algorithm to find optical flow
    p_new, status, error = cv2.calcOpticalFlowPyrLK(old_gray, fgray, p_old, p_new, **lk_params)
    
    #find the best matching points
    good_new = p_new[status == 1] #status is 1 for a match
    good_old = p_old[status == 1]
    
    mask = np.zeros_like(frame);
    
    #draw the overlaying tracking img
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel() #tmp new value
        c,d = old.ravel() #tmp old value -- necessary for drawing line tracking

        #draws a line connecting the old point with the new point
        cv2.line(mask,(a,b),(c,d),(0,255,0),2) 

        #draws the new dot
        cv2.circle(frame,(int(a),int(b)),4,(255,0,0),-1)

    #this is if we want to add lines into the tracking to follow the path
    img = cv2.add(frame,mask)
    
    #show on window
    cv2.imshow("frame",img)
    out_video.write(img)

    #update the previous frame and previous points
    old_gray = fgray.copy()
    
    #to exit window
    if cv2.waitKey(10) == 27:
        break

#clean up
cv2.destroyAllWindows()
out_video.release()
cap.release()
