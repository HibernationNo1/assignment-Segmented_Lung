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
	# 학습 후 성능 향상을 위해 image size을 조금씩 늘려볼것 (800~1024)
	IMAGE_MIN_DIM = 256
	IMAGE_MAX_DIM = 256
    
	USE_MINI_MASK = False

	# input image가 작기 때문에 anchor size도 작은걸로 사용 
	# 해당 project의 object 크기는 128을 넘어가지 않음
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



### load dataset

```python
dataset_train, dataset_validation = utils.load_dataset(trainig_config.TRAIN_DATA_RATIO, path_dataset)
```



### training

#### Create model in training mode

```python
model = MaskRCNN(mode="training", config = trainig_config, model_dir = model_dir)
```



#### Train in two stages

```python
model.train(dataset_train, dataset_validation, 
			learning_rate=trainig_config.LEARNING_RATE, 
			epochs= 2, 
			layers='heads') 
```

heads layer만 학습하는 경우

fpn, rpn, mrcnn layer 만 학습을 진행한다.

[detail]



```python
model.train(dataset_train, dataset_validation, 
			learning_rate=trainig_config.LEARNING_RATE / 5,
			epochs= 3, 
			layers="all")
```

모든 layer에 대해 학습을 진행하는 경우

backbone network인 resnet-101까지 학습을 진행한다.

[detail]





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

# model.train(dataset_train, dataset_validation, 
# 			learning_rate=trainig_config.LEARNING_RATE / 5,
# 			epochs= 7, 
# 			layers="all")
```

