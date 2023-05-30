# Cerebral-Artery-Distance-Map
<img align="right" height="800" width="400" src="https://github.com/fetouh15/Cerebral-Artery-Distance-Map/assets/38469694/8bfef7a1-6a8a-414e-8f26-cf9e9a313192"> 


## Description
This project computes the minimum Euclidean distance between each voxel in the mask and the artery of choice.
Saves the distance map (nifti image) of the artery.
## How it works?

The scripts uses the following **required** input parameters:
 + **Cerebral Arterial Tree Segmentation**
 + **Brain mask**
 + **Targeted Artery Label Number**
 + **Output Directory**  
  
  and **optional** parameters:
 + **TOF-MRA**
 + **Right Hemishphere Artery Label Number** 

The cerebral arterial tree segmentation is filtered using the target arteries label numbers. If artery is found in both hemishpere enter the left hemishpere label as the target artery and the right hemisphere in the corsponding argument parameter. The minimum Euclidean distance between each voxel in the mask and the artery of choice is computed, and saved as a distance map.


