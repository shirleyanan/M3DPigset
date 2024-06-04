# A multi-view pig 2D-3D pose dataset (M3DPigset)
  We provide a multi-view 2D-3D pig pose dataset that can be used for pig behavior analysis. This dataset was captured in various poses in natural pigpen settings using multiple low frame rate cameras. It not only enriches the available data pool but also serves as a benchmark for evaluating and advancing pig-specific 3D pose estimation techniques. The dataset contains various movements like running, walking and jumping which helps to analyze the behavior of pigs in 3D. 

<p align="center">
<img src="https://github.com/shirleyanan/M3DPigset/blob/main/images/Fig1.jpg?format=1000w?format=1000w" height="500">
<img src="https://github.com/shirleyanan/M3DPigset/blob/main/images/github.gif?format=1000w?format=1000w" height="500">
</p>

# Download
**Labeled_2D:** The labled 2D pose (4.07GB for zipflie) can be downloaded from [quark](https://pan.quark.cn/s/00d2f8ba144)(extract code:MLp5).

**Predicted_2D3D:** The predicted 2D pose (12.4GB for zipflie) with 3D can be download frome [quark](https://pan.quark.cn/s/cb81b1c0e24)(extract code:FDhr).

# Description
We defined 26 joints for pig.
</p>
<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 70%;"
    src="https://github.com/shirleyanan/M3DPigset/blob/main/images/Fig2.jpg" 
    alt="">
</img>

M3DPigset contains a total of 57 video sequences of pigs running, walking, and jumping, with a total of 8 pigs. Detailed data information is as follows:

For each pig, this data is available in the form of:

* Multi-view RGB footage recorded at 25 fps
* Partially labeled multi-view 2D joints, see the details in the description of **Labeled_2D**.There are a total of 5548 images.
* There are a total of 18168 trained [Deeplabcut](https://github.com/DeepLabCut/DeepLabCut) predicted 2D poses and corresponding 4542 3D poses, see the details in the description of **Predicted_2D3D**.

## Layout 
### **Labeled_2D**
</p>
<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 45%;"
    src="https://github.com/shirleyanan/M3DPigset/blob/main/images/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20240603180528.png" 
    alt="">
</img>

  Data for each pig is located in its own folder. `Jump1_middle2_D01` represents the `sequence_pigID_view`.`CollectedData_shirley.csv` stores the joint positions corresponding to each `Image*.jpg`.
The structure of this folder is as follows:

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

  
### **Predicted_2D3D** 
</p>
<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 45%;"
    src="https://github.com/shirleyanan/M3DPigset/blob/main/images/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20240603180544.png" 
    alt="">
</img>

1. Data for each pig is located in its own folder. The `4_View_scene.json` file under the `extrinsic_calib` folder stores the intrinsic and extrinsic parameters of 4 cameras, in the order of `cam0`, `cam1`, `cam2`, `cam3`.

2. `dlc` stores continuous frames of 2D poses from multiple viewpoints.
3. `sba.pkl` in 3D contains the corresponding 3D poses, with dimensions `[frames, 26, 3, 1]`, representing the sequence length `frames` and the 3D coordinates of `26` joints (ordered according to the **`markers`** below).

```shell
**markers**
['l_ear', 'r_ear', 'chin', 'neck_front', 'neck_back','spine_1', 'spine_6', 'l_shoulder', 'l_front_knee', 'l_front_ankle', 'l_front_paw',
            'r_shoulder', 'r_front_knee', 'r_front_ankle', 'r_front_paw', 'l_hip', 'l_back_knee', 'l_back_ankle', 'l_back_paw',
            'r_hip', 'r_back_knee', 'r_back_ankle', 'r_back_paw','tail_1', 'tail_4', 'tail_7']
```

The structure of this folder is as follows:

- dataset
  - big
    - Run3
      - dlc
        - cam0_Run3_big_D02DLC_resnet152_Multi-view-pig-jointsJun26shuffle1_1030000_filtered.csv
        - cam1_Run3_big_D01DLC_resnet152_Multi-view-pig-jointsJun26shuffle1_1030000_filtered.csv
        - ...
      - 3D
        - sba.pickle
        - ...
      - images
        - cam0
          - Image0.jpg
          - Image1.jpg
          - ...
        - ...
    - ...
  - middle1
  - ...
- extrinsic_calib
  - 4_View_scene.json
    
### Note:
In **Labeled_2D**, `D01` view corresponds to `cam1` in **Predicted_2D3D**, `D02` corresponds to `cam0`, `D03` corresponds to `cam2`, `D04` corresponds to `cam3`.

