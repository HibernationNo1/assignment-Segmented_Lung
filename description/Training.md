# Training

### import

```python
import os

import config 
from model import MaskRCNN
import utils
```





### set path dir

```python
# path of model to save 
model_dir = os.path.join(os.getcwd(), "mask-rcnn")
os.makedirs(model_dir, exist_ok=True) 

# path of dataset to load
path_dataset = os.path.join(os.getcwd(), 'training_dataset'  + '\dataset.json')
```

trained model이 있는 path와 dataset이 있는 path로부터 training에 필요한 data를 load합니다.



### set Train Config

```python
class TrainConfig(config.BaseConfig):
	"""
	BaseConfig을 상속받아 실제 받아온 data을 위한 속성값을 재정의한다.
    
	해당 project의 dataset 기준으로 값 재정의
	"""
	NAME = "lungs"
    
	
	STEPS_PER_EPOCH = 500
	VALIDATION_STEPS = 300

	POST_NMS_ROIS_TRAINING = 500
	POST_NMS_ROIS_INFERENCE = 250
    
	# 해당 poject의 dataset은 data의 개수가 적기 때문에 batch_size = 2로 맞춰본다.
	#   + 해당 poject 진행자는 3GB GPU 사용중
	IMAGES_PER_GPU = 2
	GPU_COUNT = 1
    
	# 해당project의 dataset에서 class는 left, right lungs, background
	NUM_CLASSES = 3  
    
	# 빠른 학습을 위해 작은 size의 image를 input
	IMAGE_MIN_DIM = 256
	IMAGE_MAX_DIM = 256
    
	USE_MINI_MASK = False

	# input image가 작기 때문에 anchor size도 작은걸로 사용 
	RPN_ANCHOR_SCALES = [16, 32, 64, 128, 256]
    
	# 해당project의 dataset에서 ROI가 많이 필요 없음
	TRAIN_ROIS_PER_IMAGE = 16

	LEARNING_RATE = 0.001

	# train data와 validation data간의 비율
	TRAIN_DATA_RATIO = 0.8

	RPN_TRAIN_ANCHORS_PER_IMAGE = 32

	
	MAX_GT_INSTANCES = 5


	TOP_DOWN_LAST_FILTER  = 128
```

```python
trainig_config = config.TrainConfig()
# trainig_config.display()
```

BaseConfig를 상속받으며, training에 필요한 hyper parameter를 따로 설정합니다. 



### load dataset

```python
dataset_train, dataset_validation = utils.load_dataset(trainig_config.TRAIN_DATA_RATIO, path_dataset)
```



### training

#### Create model in training mode

```python
model = MaskRCNN(mode="training", config = trainig_config, model_dir = model_dir)
```



>**load trained model**
>
>```python
>model.load_weights(model.find_last(), by_name=True)
>```
>
>이전에 학습했던 model 중 가장 마지막 model의 wieght를 가져와서 이어서 학습할 경우
>
>model instance 선언 후 load_weights를 호출한다.



#### Train in two stages

```python
model.train(dataset_train, dataset_validation, 
			learning_rate=trainig_config.LEARNING_RATE, 
			epochs= 2, 
			layers='heads') 
```

heads layer만 학습하는 경우

backbone network를 제외하고 fpn, rpn, mrcnn layer 만 학습을 진행합니다.

```
Selecting layers to train
fpn_c5p5               (Conv2D)
fpn_c4p4               (Conv2D)
fpn_c3p3               (Conv2D)
fpn_c2p2               (Conv2D)
fpn_p5                 (Conv2D)
fpn_p2                 (Conv2D)
fpn_p3                 (Conv2D)
fpn_p4                 (Conv2D)
In model:  rpn_model
    rpn_conv_shared        (Conv2D)
    rpn_class_raw          (Conv2D)
    rpn_bbox_pred          (Conv2D)
mrcnn_mask_conv1       (TimeDistributed)
mrcnn_mask_bn1         (TimeDistributed)
mrcnn_mask_conv2       (TimeDistributed)
mrcnn_mask_bn2         (TimeDistributed)
mrcnn_class_conv1      (TimeDistributed)
mrcnn_class_bn1        (TimeDistributed)
mrcnn_mask_conv3       (TimeDistributed)
mrcnn_mask_bn3         (TimeDistributed)
mrcnn_class_conv2      (TimeDistributed)
mrcnn_class_bn2        (TimeDistributed)
mrcnn_mask_conv4       (TimeDistributed)
mrcnn_mask_bn4         (TimeDistributed)
mrcnn_bbox_fc          (TimeDistributed)
mrcnn_mask_deconv      (TimeDistributed)
mrcnn_class_logits     (TimeDistributed)
mrcnn_mask             (TimeDistributed)
```





```python
model.train(dataset_train, dataset_validation, 
			learning_rate=trainig_config.LEARNING_RATE / 5,
			epochs= 3, 
			layers="all")
```

모든 layer에 대해 학습을 진행하는 경우

backbone network인 resnet-101까지 학습을 진행합니다.

```
Selecting layers to train
conv1                  (Conv2D)
bn_conv1               (BatchNorm)
res2a_branch2a         (Conv2D)
bn2a_branch2a          (BatchNorm)
res2a_branch2b         (Conv2D)
bn2a_branch2b          (BatchNorm)
res2a_branch2c         (Conv2D)
res2a_branch1          (Conv2D)
bn2a_branch2c          (BatchNorm)
bn2a_branch1           (BatchNorm)
res2b_branch2a         (Conv2D)
bn2b_branch2a          (BatchNorm)
res2b_branch2b         (Conv2D)
bn2b_branch2b          (BatchNorm)
res2b_branch2c         (Conv2D)
bn2b_branch2c          (BatchNorm)
res2c_branch2a         (Conv2D)
bn2c_branch2a          (BatchNorm)
res2c_branch2b         (Conv2D)
bn2c_branch2b          (BatchNorm)
res2c_branch2c         (Conv2D)
bn2c_branch2c          (BatchNorm)
res3a_branch2a         (Conv2D)
bn3a_branch2a          (BatchNorm)
res3a_branch2b         (Conv2D)
bn3a_branch2b          (BatchNorm)
res3a_branch2c         (Conv2D)
res3a_branch1          (Conv2D)
bn3a_branch2c          (BatchNorm)
bn3a_branch1           (BatchNorm)
res3b_branch2a         (Conv2D)
bn3b_branch2a          (BatchNorm)
res3b_branch2b         (Conv2D)
bn3b_branch2b          (BatchNorm)
res3b_branch2c         (Conv2D)
bn3b_branch2c          (BatchNorm)
res3c_branch2a         (Conv2D)
bn3c_branch2a          (BatchNorm)
res3c_branch2b         (Conv2D)
bn3c_branch2b          (BatchNorm)
res3c_branch2c         (Conv2D)
bn3c_branch2c          (BatchNorm)
res3d_branch2a         (Conv2D)
bn3d_branch2a          (BatchNorm)
res3d_branch2b         (Conv2D)
bn3d_branch2b          (BatchNorm)
res3d_branch2c         (Conv2D)
bn3d_branch2c          (BatchNorm)
res4a_branch2a         (Conv2D)
bn4a_branch2a          (BatchNorm)
res4a_branch2b         (Conv2D)
bn4a_branch2b          (BatchNorm)
res4a_branch2c         (Conv2D)
res4a_branch1          (Conv2D)
bn4a_branch2c          (BatchNorm)
bn4a_branch1           (BatchNorm)
res4b_branch2a         (Conv2D)
bn4b_branch2a          (BatchNorm)
res4b_branch2b         (Conv2D)
bn4b_branch2b          (BatchNorm)
res4b_branch2c         (Conv2D)
bn4b_branch2c          (BatchNorm)
res4c_branch2a         (Conv2D)
bn4c_branch2a          (BatchNorm)
res4c_branch2b         (Conv2D)
bn4c_branch2b          (BatchNorm)
res4c_branch2c         (Conv2D)
bn4c_branch2c          (BatchNorm)
res4d_branch2a         (Conv2D)
bn4d_branch2a          (BatchNorm)
res4d_branch2b         (Conv2D)
bn4d_branch2b          (BatchNorm)
res4d_branch2c         (Conv2D)
bn4d_branch2c          (BatchNorm)
res4e_branch2a         (Conv2D)
bn4e_branch2a          (BatchNorm)
res4e_branch2b         (Conv2D)
bn4e_branch2b          (BatchNorm)
res4e_branch2c         (Conv2D)
bn4e_branch2c          (BatchNorm)
res4f_branch2a         (Conv2D)
bn4f_branch2a          (BatchNorm)
res4f_branch2b         (Conv2D)
bn4f_branch2b          (BatchNorm)
res4f_branch2c         (Conv2D)
bn4f_branch2c          (BatchNorm)
res4g_branch2a         (Conv2D)
bn4g_branch2a          (BatchNorm)
res4g_branch2b         (Conv2D)
bn4g_branch2b          (BatchNorm)
res4g_branch2c         (Conv2D)
bn4g_branch2c          (BatchNorm)
res4h_branch2a         (Conv2D)
bn4h_branch2a          (BatchNorm)
res4h_branch2b         (Conv2D)
bn4h_branch2b          (BatchNorm)
res4h_branch2c         (Conv2D)
bn4h_branch2c          (BatchNorm)
res4i_branch2a         (Conv2D)
bn4i_branch2a          (BatchNorm)
res4i_branch2b         (Conv2D)
bn4i_branch2b          (BatchNorm)
res4i_branch2c         (Conv2D)
bn4i_branch2c          (BatchNorm)
res4j_branch2a         (Conv2D)
bn4j_branch2a          (BatchNorm)
res4j_branch2b         (Conv2D)
bn4j_branch2b          (BatchNorm)
res4j_branch2c         (Conv2D)
bn4j_branch2c          (BatchNorm)
res4k_branch2a         (Conv2D)
bn4k_branch2a          (BatchNorm)
res4k_branch2b         (Conv2D)
bn4k_branch2b          (BatchNorm)
res4k_branch2c         (Conv2D)
bn4k_branch2c          (BatchNorm)
res4l_branch2a         (Conv2D)
bn4l_branch2a          (BatchNorm)
res4l_branch2b         (Conv2D)
bn4l_branch2b          (BatchNorm)
res4l_branch2c         (Conv2D)
bn4l_branch2c          (BatchNorm)
res4m_branch2a         (Conv2D)
bn4m_branch2a          (BatchNorm)
res4m_branch2b         (Conv2D)
bn4m_branch2b          (BatchNorm)
res4m_branch2c         (Conv2D)
bn4m_branch2c          (BatchNorm)
res4n_branch2a         (Conv2D)
bn4n_branch2a          (BatchNorm)
res4n_branch2b         (Conv2D)
bn4n_branch2b          (BatchNorm)
res4n_branch2c         (Conv2D)
bn4n_branch2c          (BatchNorm)
res4o_branch2a         (Conv2D)
bn4o_branch2a          (BatchNorm)
res4o_branch2b         (Conv2D)
bn4o_branch2b          (BatchNorm)
res4o_branch2c         (Conv2D)
bn4o_branch2c          (BatchNorm)
res4p_branch2a         (Conv2D)
bn4p_branch2a          (BatchNorm)
res4p_branch2b         (Conv2D)
bn4p_branch2b          (BatchNorm)
res4p_branch2c         (Conv2D)
bn4p_branch2c          (BatchNorm)
res4q_branch2a         (Conv2D)
bn4q_branch2a          (BatchNorm)
res4q_branch2b         (Conv2D)
bn4q_branch2b          (BatchNorm)
res4q_branch2c         (Conv2D)
bn4q_branch2c          (BatchNorm)
res4r_branch2a         (Conv2D)
bn4r_branch2a          (BatchNorm)
res4r_branch2b         (Conv2D)
bn4r_branch2b          (BatchNorm)
res4r_branch2c         (Conv2D)
bn4r_branch2c          (BatchNorm)
res4s_branch2a         (Conv2D)
bn4s_branch2a          (BatchNorm)
res4s_branch2b         (Conv2D)
bn4s_branch2b          (BatchNorm)
res4s_branch2c         (Conv2D)
bn4s_branch2c          (BatchNorm)
res4t_branch2a         (Conv2D)
bn4t_branch2a          (BatchNorm)
res4t_branch2b         (Conv2D)
bn4t_branch2b          (BatchNorm)
res4t_branch2c         (Conv2D)
bn4t_branch2c          (BatchNorm)
res4u_branch2a         (Conv2D)
bn4u_branch2a          (BatchNorm)
res4u_branch2b         (Conv2D)
bn4u_branch2b          (BatchNorm)
res4u_branch2c         (Conv2D)
bn4u_branch2c          (BatchNorm)
res4v_branch2a         (Conv2D)
bn4v_branch2a          (BatchNorm)
res4v_branch2b         (Conv2D)
bn4v_branch2b          (BatchNorm)
res4v_branch2c         (Conv2D)
bn4v_branch2c          (BatchNorm)
res4w_branch2a         (Conv2D)
bn4w_branch2a          (BatchNorm)
res4w_branch2b         (Conv2D)
bn4w_branch2b          (BatchNorm)
res4w_branch2c         (Conv2D)
bn4w_branch2c          (BatchNorm)
res5a_branch2a         (Conv2D)
bn5a_branch2a          (BatchNorm)
res5a_branch2b         (Conv2D)
bn5a_branch2b          (BatchNorm)
res5a_branch2c         (Conv2D)
res5a_branch1          (Conv2D)
bn5a_branch2c          (BatchNorm)
bn5a_branch1           (BatchNorm)
res5b_branch2a         (Conv2D)
bn5b_branch2a          (BatchNorm)
res5b_branch2b         (Conv2D)
bn5b_branch2b          (BatchNorm)
res5b_branch2c         (Conv2D)
bn5b_branch2c          (BatchNorm)
res5c_branch2a         (Conv2D)
bn5c_branch2a          (BatchNorm)
res5c_branch2b         (Conv2D)
bn5c_branch2b          (BatchNorm)
res5c_branch2c         (Conv2D)
bn5c_branch2c          (BatchNorm)
fpn_c5p5               (Conv2D)
fpn_c4p4               (Conv2D)
fpn_c3p3               (Conv2D)
fpn_c2p2               (Conv2D)
fpn_p5                 (Conv2D)
fpn_p2                 (Conv2D)
fpn_p3                 (Conv2D)
fpn_p4                 (Conv2D)
In model:  rpn_model
    rpn_conv_shared        (Conv2D)
    rpn_class_raw          (Conv2D)
    rpn_bbox_pred          (Conv2D)
mrcnn_mask_conv1       (TimeDistributed)
mrcnn_mask_bn1         (TimeDistributed)
mrcnn_mask_conv2       (TimeDistributed)
mrcnn_mask_bn2         (TimeDistributed)
mrcnn_class_conv1      (TimeDistributed)
mrcnn_class_bn1        (TimeDistributed)
mrcnn_mask_conv3       (TimeDistributed)
mrcnn_mask_bn3         (TimeDistributed)
mrcnn_class_conv2      (TimeDistributed)
mrcnn_class_bn2        (TimeDistributed)
mrcnn_mask_conv4       (TimeDistributed)
mrcnn_mask_bn4         (TimeDistributed)
mrcnn_bbox_fc          (TimeDistributed)
mrcnn_mask_deconv      (TimeDistributed)
mrcnn_class_logits     (TimeDistributed)
mrcnn_mask             (TimeDistributed)
```







## Full code

```python
import os

import config 
from model import MaskRCNN
import utils


#set path dir
# path of model to save 
model_dir = os.path.join(os.getcwd(), "mask-rcnn")
os.makedirs(model_dir, exist_ok=True) 

# path of dataset to load
path_dataset = os.path.join(os.getcwd(), 'tmp_dataset'  + '\dataset.json')


trainig_config = config.TrainConfig()
# trainig_config.display()

#load dataset
dataset_train, dataset_validation = utils.load_dataset(trainig_config.TRAIN_DATA_RATIO, path_dataset)


### training
model = MaskRCNN(mode="training", config = trainig_config, model_dir = model_dir)


model.train(dataset_train, dataset_validation, 
			learning_rate=trainig_config.LEARNING_RATE, 
			epochs= 5, 
			layers='heads') 

# model.load_weights(model.find_last(), by_name=True)
# model.train(dataset_train, dataset_validation, 
# 			learning_rate=trainig_config.LEARNING_RATE / 5,
# 			epochs= 7, 
# 			layers="all")
```

