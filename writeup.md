# **Advanced-Lane-Lines** 


## 1. Pipeline description
My pipeline consists of 9 steps:</br>
1. [Camera Calibration](#camera-calibration)</br>
2. [Distortion Correction](#distortion-correction)</br>
3. [Finding Lane Lines](#finding-lane-lines)</br>
4. [Gaussian blurring and Dilation](#gaussian-blurring-and-dilation)</br>
5. [Region of interest definition](#region-of-interest-definition)</br>
6. [Perspective Transform](#perspective-transform)</br>
7. [Fitting Lane Lines](#fitting-lane-lines)</br>
8. [Radius of Curvature and Vehicle Position](#radius-of-curvature-and-vehicle-position)</br>
9. [Inverse Perspective Transform](#inverse-perspective-transform)</br>
</br>

Even before explaining the algorithm in detail, I would like to introduce few variables which control the workflow of my pipeline,
```
## Flags that control the algorithm
flagundistort=False
flagsavetemp=False
flagvideo=True
flagM=False
flagM1=False
```
* flagundistort - When Set to true, will undistort all images present in [Distorted Folder of Temp Directory](./Temp/Distorted)
* flagsavetemp  - When set to true, will save all images of [Finding Lane Lines](#finding-lane-lines)
* flagvideo     - When set to true, will run [Project Video](./test_videos/project_video.mp4) and when set to false, will Run all images
                  in [Test Image Directory](./test_images)
* flagM & flagM1 - These flags are internal ones, Used to improve performance of the pipeline.

### Camera Calibration
### Distortion Correction
### Finding Lane Lines
### Gaussian blurring and Dilation
### Region of interest definition
To filter out unnecessary objects in the image, the region of interest is defined. Such mask (here it's trapezoid) is then applied to the working image.

```python
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)     
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

### Perspective Transform
### Fitting Lane Lines
### Radius of Curvature and Vehicle Position
### Inverse Perspective Transform

