# A multi-view pig 2D-3D pose dataset (M3DPigset)
  We provide a multi-view 2D-3D pig pose dataset that can be used for pig behavior analysis. This dataset was captured in various poses in natural pigpen settings using multiple low frame rate cameras. It not only enriches the available data pool but also serves as a benchmark for evaluating and advancing pig-specific 3D pose estimation techniques. The dataset contains various movements like running, walking and jumping which helps to analyze the behavior of pigs in 3D. 

<p align="center">
<img src="https://github.com/shirleyanan/M3DPigset/blob/main/images/Fig1.jpg" width="95%">
</p>

# Download
**Labeled_2D:** The labled 2D pose (4.07GB for zipflie) can be downloaded from [quark](https://pan.quark.cn/s/00d2f8ba1447)(extract code:MLp5).

**Predicted_2D3D:** The predicted 2D pose (12.4GB for zipflie) with 3D can be download frome [quark](https://drive.google.com/drive/folders/1RC2eLC0VJ-3wMhJj90IV0IsVjZSvO2I6).

# Description
We defined 26 joints for pig.
<p align="center">
<img src="https://github.com/shirleyanan/M3DPigset/blob/main/images/Fig2.jpg" width="50%">
</p>

M3DPigset contains a total of 57 video sequences of pigs running, walking, and jumping, with a total of 8 pigs. Detailed data information is as follows:

For each pig, this data is available in the form of:

* Multi-view RGB footage recorded at 25 fps
* Partially labeled multi-view 2D joints, see the details in the description of **Labeled_2D**.There are a total of 5548 images.
* There are a total of 18168 trained [Deeplabcut](https://github.com/DeepLabCut/DeepLabCut) predicted 2D poses and corresponding 4542 3D poses, see the details in the description of **Predicted_2D3D**.

## Layout 
**Labeled_2D**
<p align="center">
<img src="https://github.com/shirleyanan/M3DPigset/blob/main/images/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20240603180528.png" width="50%">
</p>
Data for each pig is located in its own folder. The structure of this folder is as follows:

- Jump1_middle2_D01
  - CollectedData_shirley.csv
  - CollectedData_shirley.h5
  - Image3.jpg
  - ...
- Jump1_middle2_D02
  - CollectedData_shirley.csv
  - CollectedData_shirley.h5
  - Image3.jpg
  - ...
- ...

  
**Predicted_2D3D** 
<p align="center">
<img src="https://github.com/shirleyanan/M3DPigset/blob/main/images/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20240603180544.png" width="50%">
</p>
Data for each pig is located in its own folder. The `4_View_scene.json` file under the 'extrinsic_calib' folder stores the intrinsic and extrinsic parameters of 4 cameras, in the order of 'cam0', 'cam1', 'cam2', 'cam3'.


The structure of this folder is as follows:

- dataset
  - 
  - CollectedData_shirley.h5
  - Image3.jpg
  - ...
- extrinsic_calib
  - 4_View_scene.json
### Note:
In **Labeled_2D**, 'D01' view corresponds to 'cam1' in **Predicted_2D3D**, 'D02' corresponds to 'cam0', 'D03' corresponds to 'cam2', 'D04' corresponds to 'cam3'.

