import numpy as np

class BaseConfig():
	"""
	model의 hyper parameters 및 flag, config값을 정의하기 위한 class
    
	train, validation 및 test간에는 해당 class를 상속하는 하위 class를 만들어 
	속성을 재 정의 해야 한다.
    
	해당 class의 모든 값은 original code의 값 그대로 유지한다.
	"""
		
	# 사용할 GPU 개수. CPU사용 시 1로 set
	GPU_COUNT = 1

	# 각 GPU마다 학습할 image의 개수. 12GB GPU기준 개당 1024×1024 image 2개 학습 가능
	IMAGES_PER_GPU = 2
    
    
	# Max number of final detections
	DETECTION_MAX_INSTANCES = 100
    
	# epoch당 training step의 수
	# tensorboard는 1 epoch마다 updata를 진행하고 model을 save한다.
	STEPS_PER_EPOCH = 1000
   	
	# validation을 진행할 step의 수
	VALIDATION_STEPS = 50

	# model의 input으로 사용하기 위해 resize한 image의 size
	# 최소값 : IMAGE_MIN_DIM, 최대값 : IMAGE_MAX_DIM
	IMAGE_MIN_DIM = 800
	IMAGE_MAX_DIM = 1024
	IMAGE_CHANNEL_COUNT = 3

	# image의 크기를 결정하는 scale. 만약 2라면, MIN_IMAGE_DIM이 없더라도 2배의 size로 resize됨
	IMAGE_MIN_SCALE = 0

	# How many anchors per image to use for RPN training
	RPN_TRAIN_ANCHORS_PER_IMAGE = 256

	# Maximum number of ground truth instances to use in one image
	MAX_GT_INSTANCES = 100
    
	# dataset에 속한 class의 개수 중 실제 classification을 진행할 class의 개수
	# (including background)
	NUM_CLASSES = 1	# Override in sub-classes
    

	# batch normalization layer를 train할지, 동결시킬지 결정하는 속성 
	#     None: Train BN layers. This is the normal mode
	#     False: Freeze BN layers. Good when using a small batch size
	#     True: (don't use). Set layer in training mode even when predicting
	TRAIN_BN = False
    
    
	# Backbone network architecture
	BACKBONE = "resnet101"
    
	# subsampled_ratio
	BACKBONE_STRIDES = [4, 8, 16, 32, 64] 
    
    
	# FPN의 Top-Down과정에서 만들어지는 M계층 feature pyramid의 channel 개수
	TOP_DOWN_PYRAMID_SIZE  = 256

      
	# RPN Anchor의 stride
	# RPN_ANCHOR_STRIDE = 1 인 경우 backbone network의 각 cell당 anchor 생성
	# RPN_ANCHOR_STRIDE = 2 인 경우 
	# anchors are created for every other cell, and so on.?? 이해가 안감
	RPN_ANCHOR_STRIDE = 1

	IMAGE_RESIZE_MODE = "square"

	# Image mean (RGB)
	MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    
	# RPN Anchor의 shape 비율  width/height의 결과값
	RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
	# RPN Anchor의 크기
	RPN_ANCHOR_SCALES = [32, 64, 128, 256, 512]
    
	# Non-max suppression을 하기 전 tf.nn.top_k으로 뽑은 ROI의 최대 개수
	PRE_NMS_LIMIT = 6000
    
	# RPN proposals을 위한 Non-max suppression의 threshold 
	RPN_NMS_THRESHOLD = 0.7
    
	# RPN Anchor에 delta를 적용시키기 위한 standard deviation 조정값
	RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    
	# final detections에서 Anchor에 delta를 적용시키기 위한 standard deviation 조정값
	BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    
	# training mode에서 NMS이후 사용할 ROI의 개수
	POST_NMS_ROIS_TRAINING = 2000
	# inference mode에서 NMS이후 사용할 ROI의 개수
	POST_NMS_ROIS_INFERENCE = 1000
    
	# training을 위해 RPN ROIs을 사용할지 여부
	# ROI가 RPN에 의해 만들어진 것이 아닌, code로 인해 사용자가 임의로 만들었을 경우 `False`
	USE_RPN_ROIS = True
    
	# ROI Pooling에 의해 만들어진 feature map의 size
	POOL_SIZE = 7
    
	# mask image에 대한 ROI Pooling으로 만들어진 feature map의 size
	MASK_POOL_SIZE = 14
    
	# image 1개당 classifier/mask heads에 전달할 ROI의 개수
	# RPN NMS threshold을 조정하면 해당 속성값을 증가시킬 수 있음
	TRAIN_ROIS_PER_IMAGE = 200
    
	# 전체 ROI중 training을 위해 classifier/mask heads에 전달할 positive ROI의 비율
	ROI_POSITIVE_RATIO = 0.33
       
    
	# Size of the fully-connected layers in the classification graph
	# RCNN paper uses 512
	FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    
	# Shape of output mask
	# MASK_SHAPE 값 변경시 neural network mask branch 로 변경해야함
	MASK_SHAPE = [28, 28]
    
	# Minimum probability value to accept a detected instance
	# 해당 속성값 미만의 ROI는 버려진다.
	DETECTION_MIN_CONFIDENCE = 0.7
    
	# Non-maximum suppression threshold for detection
	DETECTION_NMS_THRESHOLD = 0.3

	LEARNING_RATE = 0.001
	LEARNING_MOMENTUM = 0.9

	# Gradient norm clipping
	GRADIENT_CLIP_NORM = 5.0

	# Weight decay regularization
	WEIGHT_DECAY = 0.0001

	# Loss weights for more precise optimization.
	# Can be used for R-CNN training setup.
	LOSS_WEIGHTS = {
		"rpn_class_loss": 1.,
		"rpn_bbox_loss": 1.,
		"mrcnn_class_loss": 1.,
		"mrcnn_bbox_loss": 1.,
		"mrcnn_mask_loss": 1.
	}

	def __init__(self):
		"""
		Set values of computed attributes.
		"""

		# set BATCH_SIZE
		self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

		# IMAGE_META_SIZE
		# image의 meta data를 담은 container의 shape
		self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

		# model의 input으로 사용하기 위해 resize한 image의 shape
		self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
									 self.IMAGE_CHANNEL_COUNT])




	def display(self):
		"""
		Display Configuration values.
		"""
		print("\nConfigurations:")
		for a in dir(self):
			if not a.startswith("__") and not callable(getattr(self, a)):
				print(f"{a:30} {getattr(self, a)}")
		print("\n")



class TrainConfig(BaseConfig):
	"""
	BaseConfig을 상속받아 실제 받아온 data을 위한 속성값을 재정의한다.
    
	해당 project의 dataset 기준으로 값 재정의
	"""
	NAME = "lungs"
    
	
	STEPS_PER_EPOCH = 300
	VALIDATION_STEPS = 200

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
	# 해당 project의 object 크기는 128을 넘어가지 않음
	RPN_ANCHOR_SCALES = [16, 32, 64, 128, 256]
    
	# 해당project의 dataset에서 ROI가 많이 필요 없음
	TRAIN_ROIS_PER_IMAGE = 64

	LEARNING_RATE = 0.001

	# train data와 validation data간의 비율
	TRAIN_DATA_RATIO = 0.9

	# anchor가 많으면 
	# anchor 적어도 충분히 detection하는데 무리 없음
	RPN_TRAIN_ANCHORS_PER_IMAGE = 128

	
	MAX_GT_INSTANCES = 5


	TOP_DOWN_LAST_FILTER  = 128

class InferenceConfig(TrainConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# for test
	TRAIN_DATA_RATIO = 1

	DETECTION_MAX_INSTANCES = 20

	SAVE_RESULT = True
