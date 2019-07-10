"""
Created on Mon Jun 24 19:09:32 2019

@author: Ascalon
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import time
import os

## Flags that control the algorithm
flagundistort=False
flagsavetemp=False
flagvideo=True
flagM=False
flagM1=False

##Hyper Parameters
LowerThres=50  
DestThres=100
M=[]
M1=[]
nwindows = 9
margin = 100
minpix = 50
src=[]
dst=[]
left_fit=[]
right_fit=[]

##Functions
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
   
    if orient=='x':
        sobel_thresh=cv2.Sobel(img,cv2.CV_64F,1,0,sobel_kernel)      
    else:
        sobel_thresh=cv2.Sobel(img,cv2.CV_64F,0,1,sobel_kernel)
    abs_sobel=np.absolute(sobel_thresh)
    scaled_sobel=np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary=np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel>=thresh[0])&(scaled_sobel<=thresh[1])]=1
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,sobel_kernel)  
    sobely=cv2.Sobel(img,cv2.CV_64F,0,1,sobel_kernel)
    abs_sobel=np.sqrt((sobelx**2)+(sobely**2))
    scalefactor=np.max(abs_sobel)/255
    abs_sobel=(abs_sobel/scalefactor).astype(np.uint8)
    mag_binary=np.zeros_like(abs_sobel)
    mag_binary[(abs_sobel>=mag_thresh[0])&(abs_sobel<=mag_thresh[1])]=1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,sobel_kernel)  
    sobely=cv2.Sobel(img,cv2.CV_64F,0,1,sobel_kernel)
    abs_sobelx=np.absolute(sobelx)
    abs_sobely=np.absolute(sobely)
    dir_sobel=np.arctan(abs_sobely/abs_sobelx)
    dir_binary=np.zeros_like(dir_sobel)
    dir_binary[(dir_sobel>=thresh[0])&(dir_sobel<=thresh[1])]=1
    return dir_binary
   
def hls_select(img, thresh=(0, 255)):
    
    imgs=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    s=imgs[:,:,2]
    hls_binary=np.zeros_like(s)
    hls_binary[(s>thresh[0])&(s<=thresh[1])]=1  
    return hls_binary
    
def hsv(img):
    
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow1 = np.array([ 0,120, 120], dtype=np.uint8)
    yellow2 = np.array([ 100,255, 255,], dtype=np.uint8)
    yellow = cv2.inRange(img, yellow1, yellow2)
    white1 = np.array([0, 0, 200], dtype=np.uint8)
    white2 = np.array([255, 30, 255], dtype=np.uint8)
    white = cv2.inRange(img1, white1, white2)
    out=cv2.bitwise_and(img, img, mask=(yellow | white))
    out=cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
    im_bw = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY)[1]
    return im_bw/255

def save_img(gradx,grady,combined,hls_binary,dir_binary,mag_binary,color):
    
    cv2.imwrite('Temp/Sobelx'+'.jpg',gradx*255)
    cv2.imwrite('Temp/Sobely'+'.jpg',grady*255)
    cv2.imwrite('Temp/combined'+'.jpg',combined*255)
    cv2.imwrite('Temp/hls_binary'+'.jpg',hls_binary*255)
    cv2.imwrite('Temp/dir_binary'+'.jpg',dir_binary*255)
    cv2.imwrite('Temp/mag_binary'+'.jpg',mag_binary*255) 
    cv2.imwrite('Temp/color'+'.jpg',color*255)
    
def color_mag_dir_thresh(img): 
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gradx=abs_sobel_thresh(gray,'x',3,(40,160))
    grady=abs_sobel_thresh(gray,'y',3,(40,160))
    mag_binary=mag_thresh(gray,3,(80,180))
    dir_binary = dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2))
    hls_binary=hls_select(img,(150,240))
    color=hsv(img)
    combined = np.zeros_like(color)
    combined[((gradx == 1) & (grady == 1)) | (color==1)| (( mag_binary==1) &  ( dir_binary==1)) |( hls_binary==1) ]=1
    if flagsavetemp:
        save_img(gradx,grady,combined,hls_binary,dir_binary,mag_binary,color)

    return combined    

    
def p_transform(img,H,W,LO,DO,M,flagM,src,dst):

    if not(flagM):
        src=np.float32([[W/2.28,H/1.6],[W/1.77,H/1.6],[W/1.2,H-LO],[W/5.73,H-LO]])
        dst=np.float32([[0+DO/2,0],[W-DO/2,0],[W-2*DO,H],[0+2*DO,H]])
        M = cv2.getPerspectiveTransform(src, dst)
        flagM=True
    warped = cv2.warpPerspective(img, M, (W,H), flags=cv2.INTER_LINEAR)
    return warped,M,src,dst,flagM

 
def find_lanes_window(warped):
    
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    window_height = np.int(warped.shape[0]//nwindows)
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds=[]
    right_lane_inds=[]
    for window in range(nwindows):
        win_y_low=warped.shape[0]-(window+1)*window_height
        win_y_high=warped.shape[0]-window*window_height
        win_xleft_low=leftx_current-margin
        win_xleft_high=leftx_current+margin
        win_xright_low=rightx_current-margin
        win_xright_high=rightx_current+margin
        good_left_inds =((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox <= win_xleft_high)).nonzero()[0]
        good_right_inds=((nonzeroy >= win_y_low)&(nonzeroy < win_y_high) &
                         (nonzerox>=win_xright_low)&(nonzerox<=win_xright_high)).nonzero()[0]
                         
        if len(good_left_inds)>minpix:
            leftx_current=np.int(np.mean(nonzerox[good_left_inds]))          
        if len(good_right_inds)>minpix:
            rightx_current=np.int(np.mean(nonzerox[good_right_inds]))
     
        left_lane_inds.append(good_left_inds) 
        right_lane_inds.append(good_right_inds)
        
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        
        pass
    
    leftx=nonzerox[left_lane_inds]
    lefty=nonzeroy[left_lane_inds]
    rightx=nonzerox[right_lane_inds]
    righty=nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return leftx,lefty,rightx,righty,left_fit,right_fit

def find_lanes_search(warped,left_fit,right_fit):
    margin = 50
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return leftx,lefty,rightx,righty,left_fit,right_fit
    
def draw_lanes(warped,M,src,dst,flagM1,M1,left_fit,right_fit):
    
    if len(left_fit) and len(right_fit):
#        leftx,lefty,rightx,righty,left_fit,right_fit=find_lanes_search(warped,left_fit,right_fit)
        leftx,lefty,rightx,righty,left_fit,right_fit=find_lanes_window(warped)
    else:   
        leftx,lefty,rightx,righty,left_fit,right_fit=find_lanes_window(warped)
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    mask=np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((mask, mask, mask))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255,0,0), thickness=15)
    
    if not(flagM1):
        M1 = cv2.getPerspectiveTransform(dst, src)
        flagM1=True
    newwarp = cv2.warpPerspective(color_warp, M1, (warped.shape[1], warped.shape[0]))
    curve,distance=CalculateRadiusOfCurvature(warped,left_fit,right_fit)
    cv2.putText(newwarp,"Radius of Curvature is " + str(int(curve))+ "m", (100,100), 2, 1, (255,255,0),2)
    cv2.putText(newwarp,"Distance from center is {:2f}".format(distance)+ "m", (100,150), 2, 1, (255,255,0),2)
    return newwarp,flagM1,M1,left_fit,right_fit
    
def CalculateRadiusOfCurvature(binary_warped,left_fit,right_fit):
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    positionCar= binary_warped.shape[1]/2
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)    
    y_eval=np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])   
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    actualPosition= (left_lane_bottom+ right_lane_bottom)/2
    distance= (positionCar - actualPosition)* xm_per_pix
    return (left_curverad + right_curverad)/2, distance
   
def weighted_img(img, initial_img, α=1, β=1, γ=0):
 
    return cv2.addWeighted(initial_img, α, img, β, γ)
    
## Camera Calibration##
def findcampar(): 

    if len(glob.glob('Temp/CameraCalibration.p')):
        dist_mtx=pickle.load(open("Temp/CameraCalibration.p","rb"))
        mtx=dist_mtx[1]
        dist=dist_mtx[2]
    else:
        PIK='Temp/CameraCalibration.p'
        imgpoints=[]
        objpoints=[]
        objp=np.zeros((9*6,3),np.float32)
        objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
        images=glob.glob('camera_cal/calibration*.jpg')
        for image in images:
            img=mpimg.imread(image)
            gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            if ret== True:
                image=image.split('\\')
                imgpoints.append(corners)
                objpoints.append(objp)
                img=cv2.drawChessboardCorners(img,(9,6),corners,ret)
                cv2.imwrite('camera_cal/chessboard/'+image[1],img)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) 
        with open(PIK, "wb") as f:
            data=[ret, mtx, dist, rvecs, tvecs]
            pickle.dump(data, f)
            
    return mtx,dist
        
def undistortimg():
    
    images=glob.glob('Temp/Distorted/*.jpg')
    if len(images)>0:
        f,ax=plt.subplots(len(images),2)
        f.tight_layout()
        for i,image in enumerate(images):
            img=mpimg.imread(image)
            image=image.split('\\')
            undimg=cv2.undistort(img, mtx, dist, None, mtx)
            cv2.imwrite('Temp/Undistorted/Undistorted_'+image[1],undimg)       
            ax[i][0].imshow(img)
            ax[i][0].set_title('Original Image', fontsize=20)
            ax[i][1].imshow(undimg)
            ax[i][1].set_title('Undistorted Image', fontsize=20)
            
def undistort(img,mtx,dist):
    
    undimg=cv2.undistort(img, mtx, dist, None, mtx)
    return undimg
 
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)     
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    
def gaussian_blur(img, kernel_size):
   
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
## Algorithm     

mtx,dist=findcampar()

if flagundistort:
    undistortimg()
    
if flagvideo:
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    cap=cv2.VideoCapture('test_videos/project_video.mp4')
    W = int(cap.get(3) )
    H = int(cap.get(4))
    video = cv2.VideoWriter('output_videos/Annotated_'+'project_video'+'.mp4', fourcc, cap.get(5), (W,H))
    start=time.time()
    midx=W/2
    maxt=H/1.6
    kernel = np.ones((3,3),np.uint8)
    vertices = np.array([[(30,H),(midx-100, maxt), (midx+100, maxt), (W-30,H)]], dtype=np.int32)
    while True:
        ret,img=cap.read()
        
        if ret:
            img=undistort(img,mtx,dist)
            combined=color_mag_dir_thresh(img)
            combined=gaussian_blur(combined,3)
            combined=cv2.dilate(combined,kernel,iterations = 1)
            masked_image=region_of_interest(combined,vertices)
            # Perspective Transform
            warped,M,src,dst,flagM=p_transform(masked_image,H,W,LowerThres,DestThres,M,flagM,src,dst)
            color_warp,flagM1,M1,left_fit,right_fit=draw_lanes(warped,M1,src,dst,flagM1,M1,left_fit,right_fit)
            out = weighted_img(img, color_warp, 0.6, 1, 0) 
            video.write(out)
            cv2.imshow('Output',out)
            k=cv2.waitKey(1)
            if k==ord('q'):
                break
        else:
            break
    end=time.time()
    print("Time:",end-start)
    cap.release()
    video.release()
    cv2.destroyAllWindows()
            
else:
    files=glob.glob(os.path.join('test_images/','*.jpg'))
    for i in files:
        left_fit=[]
        right_fit=[]
        tmpname=str.split(i,'\\')
        tmpname=str.split(tmpname[1],'.')
        img=cv2.imread(i)
        img=undistort(img,mtx,dist)
        H=img.shape[0]
        W=img.shape[1]
        midx=W/2
        maxt=H/1.6
        kernel = np.ones((3,3),np.uint8)
        vertices = np.array([[(30,H),(midx-100, maxt), (midx+100, maxt), (W-30,H)]], dtype=np.int32)
        combined=color_mag_dir_thresh(img) 
        combined=gaussian_blur(combined,3)
        combined=cv2.dilate(combined,kernel,iterations = 1)
        masked_image=region_of_interest(combined,vertices)
        warped,M,src,dst,flagM=p_transform(combined,H,W,LowerThres,DestThres,M,flagM,src,dst)
        color_warp,flagM1,M1,left_fit,right_fit=draw_lanes(warped,M1,src,dst,flagM1,M1,left_fit,right_fit)
        out = weighted_img(img, color_warp, 0.6, 1, 0) 
        cv2.imwrite('output_images/Annotated_'+tmpname[0]+'.jpg',out)


   
