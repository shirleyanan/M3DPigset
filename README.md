# A multi-view pig 2D-3D pose dataset (M3DPigset)
  We provide a multi-view 2D-3D pig pose dataset that can be used for pig behavior analysis. This dataset was captured in various poses in natural pigpen settings using multiple low frame rate cameras. It not only enriches the available data pool but also serves as a benchmark for evaluating and advancing pig-specific 3D pose estimation techniques. The dataset contains various movements like running, walking and jumping which helps to analyze the behavior of pigs in 3D. 

<p align="center">
<img src="https://github.com/shirleyanan/M3DPigset/blob/main/images/Fig1.jpg" width="95%">
</p>

# Download
**Labeled_2D:** The labled 2D pose (4.07GB for zipflie) can be downloaded from [quark](https://pan.quark.cn/s/00d2f8ba1447)(extract code:MLp5).

**Predicted_2D3D:** The predicted 2D pose (12.4GB for zipflie) with 3D can be download frome [quark](https://drive.google.com/drive/folders/1RC2eLC0VJ-3wMhJj90IV0IsVjZSvO2I6).

# Description
M3DPigset contains a total of 57 video sequences of pigs running, walking, and jumping, with a total of 8 pigs. Detailed data information is as follows:

For each dog, this data is available in the form of:

* multi-view RGB footage recorded at 25 fps
* Partially labeled multi-view 2D joints, see the details in the description of **Labeled_2D**.



## Joints
We defined 26 joints for pig.
<p align="center">
<img src="https://github.com/shirleyanan/M3DPigset/blob/main/images/Fig2.jpg" width="95%">
</p>

## **Labeled_2D**
<p align="center">
<img src="https://github.com/shirleyanan/M3DPigset/blob/main/images/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20240603180528.png" width="95%">
</p>


## **Predicted_2D3D** 
The 2D pose situation predicted by the trained [Deeplabcut](https://github.com/DeepLabCut/DeepLabCut).
<p align="center">
<img src="https://github.com/shirleyanan/M3DPigset/blob/main/images/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20240603180544.png" width="95%">
</p>
