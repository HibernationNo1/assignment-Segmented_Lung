# Inference

png 또는 jpg 형식의 file을 input으로 받은 후 trained model에 의한 segmentation을 진행한 후 결과에 대해 window로 출력 또는 png 형식의 file로 save하는 code입니다.



### import

```python
import os
import cv2
import glob

import visualize
import utils
import config 
from model import MaskRCNN
```



### set Inference Config

```python
class InferenceConfig(TrainConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# for test
	TRAIN_DATA_RATIO = 1

	DETECTION_MAX_INSTANCES = 20

	# True : 여러 model을 사용하여 inference 후 ensemble bagging
	# False : 단일 model을 사용하여 inference
	ENSEMBLE = True

	# load 할 model의 개수
	NUMBER_OF_MODEL = 5

	# True : .png 형식의 file로 비교 image save
	# False : plt.show()를 통해 window창으로 image 시각화
	SAVE_RESULT = True
```

- `inference_config.SAVE_RESULT = False` 

  infetence 결과를 `plt.show()` 를 통해 window로 보여줍니다.

- `inference_config.SAVE_RESULT = True`

   infetence 결과에 대해 original input image와 비교하여 png형식의 file로 저장합니다.

  

```python
inference_config = InferenceConfig()
# inference_config.display()
```



### set path directory

```python
# path of model to load weights
model_dir = os.path.join(os.getcwd(), "model_mask-rcnn")
# path of image to inference
path_dataset = os.path.join(os.getcwd() , 'sample_test_image')

if inference_config.SAVE_RESULT:
	path_result = os.path.join(os.getcwd() , 'result_inference')
	os.makedirs(path_result, exist_ok=True)
```





### Create model in inference mode and load weights

```python
model_list = list()
if inference_config.ENSEMBLE: 
	for i in range(inference_config.NUMBER_OF_MODEL):
		# Create model in inference mode
		model = MaskRCNN(mode="inference", 
							config=inference_config,
							model_dir=model_dir)
		# load weights
		model_path = model.find_last(file_name = "_" + str(i))
		model.load_weights(model_path, by_name=True)
		# model_list [model_0, model_1, ...]
		model_list.append(model)
else : 
	model = MaskRCNN(mode="inference", 
							config=inference_config,
							model_dir=model_dir)
	model_path = model.find_last()
	model.load_weights(model_path, by_name=True)
	model_list.append(model)
```

- `inference_config.ENSEMBLE = True` 

  여러 model을 통해 inference 결과를 얻고 집계하는 ensemble - bagging 기법을 사용합니다.

- `inference_config.ENSEMBLE = False` 

  단일 model을 통해 inference 결과를 얻습니다.





### detect and visualize

```python
for iter, path_ in enumerate(sorted(glob.glob (path_dataset + '\*.*'))):	
	title = path_.split("\\")[-1]
	original_image = cv2.imread(path_)	# <class 'numpy.ndarray'>
	original_image = utils.preprocessing_HE(original_image)

	print(f"file name : {title}")

	result_list = list()

	verbose = 0
	for i in range(len(model_list)):
		if i == len(model_list) - 1:
			verbose = 1

		results = model_list[i].detect([original_image], verbose=verbose)
		# result_list = [results_1, results_2, results_3]
		result_list.append(results)

	# results = ["rois" : [num_rois, (y1, x1, y2, x2)], 
	#			 "class_ids" : [num_rois]
	#			 "scores": [num_rois]
	#  			 "masks": [H, W, num_rois] 


	visualize.display_instances(original_image, 
								len(model_list),
								result_list, 
								title,
								save = inference_config.SAVE_RESULT, path = path_result)
```

image Ensemble, Morphology, Labeling 연산은 [visualize.py](https://github.com/HibernationNo1/project_segmentation_lungs/blob/master/code/mask_rcnn/visualize.py)에서 확인하실 수 있습니다.





## Full code

```python
import os
import cv2
import glob

import visualize
import utils
import config 
from model import MaskRCNN

inference_config = config.InferenceConfig()
# inference_config.display()


## set path directory
# path of model to load weights
model_dir = os.path.join(os.getcwd(), "model_mask-rcnn")
# path of image to inference
path_dataset = os.path.join(os.getcwd() , 'test_image')

if inference_config.SAVE_RESULT:
	path_result = os.path.join(os.getcwd() , 'result_inference')
	os.makedirs(path_result, exist_ok=True)  


model_list = list()
if inference_config.ENSEMBLE: 
	for i in range(inference_config.NUMBER_OF_MODEL):
		# Create model in inference mode
		model = MaskRCNN(mode="inference", 
							config=inference_config,
							model_dir=model_dir)
		# load weights
		model_path = model.find_last(file_name = "_" + str(i))
		model.load_weights(model_path, by_name=True)
		# model_list [model_0, model_1, ...]
		model_list.append(model)
else : 
	model = MaskRCNN(mode="inference", 
							config=inference_config,
							model_dir=model_dir)
	model_path = model.find_last()
	model.load_weights(model_path, by_name=True)
	model_list.append(model)


for iter, path_ in enumerate(sorted(glob.glob (path_dataset + '\*.*'))):	
	title = path_.split("\\")[-1]
	original_image = cv2.imread(path_)	# <class 'numpy.ndarray'>
	original_image = utils.preprocessing_HE(original_image)

	print(f"file name : {title}")

	result_list = list()

	verbose = 0
	for i in range(len(model_list)):
		if i == len(model_list) - 1:
			verbose = 1

		results = model_list[i].detect([original_image], verbose=verbose)
		# result_list = [results_1, results_2, results_3]
		result_list.append(results)

	# results = ["rois" : [num_rois, (y1, x1, y2, x2)], 
	#			 "class_ids" : [num_rois]
	#			 "scores": [num_rois]
	#  			 "masks": [H, W, num_rois] 


	visualize.display_instances(original_image, 
								len(model_list),
								result_list, 
								title,
								save = inference_config.SAVE_RESULT, path = path_result)
```

