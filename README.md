# Mask R-CNN for Lung Segmentation

서울대학병원 융합의학과 김영곤 교수님의 과제 수행 repository입니다.

흉부 CT image로부터 letf, right lung을 구분하여 segmentation을 진행하는 모델을 구현하는 project입니다.

1. pre trained model을 통해 흉부 CT image로부터 좌, 우 구분 없는 lung mask에 대한 data를 확보합니다.
2. 확보한 data로부터 좌, 우 lung의 구분을 진행하여 새로운 dataset을 만들어 json형식의 file로 저장합니다.
3. json file을 load하여 좌, 우 lung의 구분이 가능한 segmentation 학습을 진행할 수 있도록 Mask R-CNN model을 설계합니다.



- 학습에 필요한 dataset은 서울대학병원 융합의학과 김영곤 교수님의 [Research-Segmentation-Lung](https://github.com/younggon2/Research-Segmentation-Lung-CXR-COVID19)를 인용하여 코드의 수정을 통해  json file로 저장하도록 했습니다.
- 학습을 진행하는 model은  https://github.com/akTwelve/Mask_RCNN 로부터 수정을 통해 학습하고자 하는 dataset에 알맞도록 구현하였습니다.
- training, test dataset source
  - [chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
  - [covid19-image-dataset](https://www.kaggle.com/pranavraikokte/covid19-image-dataset)






## Description

- [Create_Dataset.dm](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/description/Create%20Dataset.md)

  input image로부터 Mask R-CNN을 학습시키기 위한 label을 담고 있는 dataset 만드는 code에 대한 설명입니다.

- [Training.md](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/description/Training.md)

  흉부 CT dataset을 통해 Mask R-CNN의 학습을 진행하는 code에 대한 설명입니다.

- [Inference.md](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/description/Inference.md)

  학습된 모델을 통해 test dataset에 대해서 inference를 진행하는 code에 대한 설명입니다.



## Code

- [create_dataset.py](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/code/create_dataset/create_dataset.py)

  training 및 test dataset을 만들어내는 code입니다.

- [model.py](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/code/mask_rcnn/model.py), [utils.py](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/code/mask_rcnn/utils.py), [config.py](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/code/mask_rcnn/config.py)

  Mask R-CNN 구현한 code입니다.

  



## Segmentation result

Segmentation inference result

![](https://github.com/HibernationNo1/segmentation_lungs/blob/master/image/r_1.png?raw=true)

![](https://github.com/HibernationNo1/segmentation_lungs/blob/master/image/r_2.png?raw=true)





### Loss

loss는 tenserboard를 사용해 시각화 했습니다.

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/loss.png?raw=true)





![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/loss_1.png?raw=true)



### Model

model을 학습하는데 사용한 dataset은 아래와 같습니다.

['Covid-19 Image Dataset'](https://www.kaggle.com/pranavraikokte/covid19-image-dataset)

70개의 image

\+

['Chest X-Ray Images'](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
300개의 image

\+

['Covid Patients Chest X-Ray'](https://www.kaggle.com/ankitachoudhury01/covid-patients-chest-xray)

147개의 image

총 547개의 image를 사용해 training dataset을 구성했습니다.



[**trained model**](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/model_mask-rcnn/lungs_model/mask_rcnn__lungs_0000.h5)

총 69개의 test image중

- 94%의 비율로(65개) left, right lung을 성공적으로 detection하고,  위 result image의 수준으로 segmentation을 진행함을 확인했습니다.

-  4% 의 비율로 (4개) left, right lung 중 1개의 instance를 detection하고 segmentation을 진행함을 확인했습니다.

  ![](https://github.com/HibernationNo1/segmentation_lungs/blob/master/image/r_3.png?raw=true)

- 1.4%의 비율로(1개) object detection에 실패함을 확인했습니다. 

  > inference과정에서 input image에 preprocessing을 적용하여 성공확률 높히기

> 해당 model과 sample test image을 통해 바로 segmentation을 진행할 수 있는 code를 ipynb file로 업로드 하기 위해 version문제를 해결하고 있습니다.





## Requirements

### create_dataset.py

#### tools

| name   | version | note             |
| ------ | ------- | ---------------- |
| CUDA   | 11.0    | cudart64_110.dll |
| cuDNN  | 8.2     | cudnn64_8.dll    |
| python | 3.8.8   |                  |

#### package

| name                | version |
| ------------------- | ------- |
| segmentation_models | 1.0.1   |
| cv2                 | 4.5.3   |
| scipy               | 1.6.2   |
| numpy               | 1.21.2  |



### Mask RCNN

#### tools

| name   | version | note             |
| ------ | ------- | ---------------- |
| CUDA   | 10.0    | cudart64_100.dll |
| cuDNN  | 7.6.4   | cudnn64_7.dll    |
| python | 3.7.11  |                  |

#### package

| name             | version |
| ---------------- | ------- |
| tensorflow       | 2.0.0   |
| tensorflow.keras | 2.2.4   |
| h5py             | 2.10.0  |
| cv2              | 4.5.3   |
| skimage          | 0.16.2  |



