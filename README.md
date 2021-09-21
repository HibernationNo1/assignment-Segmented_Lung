# Mask R-CNN for Lung Segmentation

서울대학병원 융합의학과 김영곤 교수님의 과제 수행 repository입니다.

해당 project에서는 흉부 CT사진을 데이터 삼아 좌, 우 lung을 구분하여 segmentation을 진행하는 모델을 구현했습니다.

- 학습에 필요한 dataset은 서울대학병원 융합의학과 김영곤 교수님의 [Research-Segmentation-Lung](https://github.com/younggon2/Research-Segmentation-Lung-CXR-COVID19)를 인용하여 코드의 수정을 통해  json file로 저장하도록 했습니다.

  [detail]

- 학습을 진행하는 model은  https://github.com/akTwelve/Mask_RCNN 의 동작을 이해하고 dataset에 알맞게 수정을 거쳐 설계하였습니다.

  [detail]



## Description

- [Create Dataset.dm] 

  input image data로부터 Mask R-CNN을 학습시키기 위한 label을 담고 있는 dataset 만드는 code에 대한 설명입니다.

- [Training.md]

  흉부 CT dataset을 통해 Mask R-CNN의 학습을 진행하는 code에 대한 설명입니다.

- [Inference.md]

  학습된 모델을 통해 test dataset에 대해서 inference를 진행하는 code에 대한 설명입니다.

- [model.py, utils.py, config.py]

  Mask R-CNN 구현한 code입니다.





## Segmentation result





## Requirements

### make_dataset

#### tools

| name  | version | note             |
| ----- | ------- | ---------------- |
| CUDA  | 11.0    | cudart64_110.dll |
| cuDNN | 8.2     | cudnn64_8.dll    |

#### package

| name                | version |
| ------------------- | ------- |
| segmentation_models | 1.0.1   |
| cv2                 | 4.5.3   |
| scipy.ndimage       | 2.0     |
| numpy               | 1.21.2  |



### training and inference

#### tools

| name  | version | note             |
| ----- | ------- | ---------------- |
| CUDA  | 10.0    | cudart64_100.dll |
| cuDNN | 7.6.4   | cudnn64_7.dll    |

#### package

| name             | version |
| ---------------- | ------- |
| tensorflow       | 2.0.0   |
| tensorflow.keras | 2.2.4   |
| h5py             | 2.10.0  |
| numpy            | 1.21.2  |
| cv2              | 4.5.3   |



