# Inference

### import

```python
import os

import visualize
import utils
import config 
from model import load_image_gt, MaskRCNN
```



### set Inference Config

```python
class InferenceConfig(config.TrainConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# for test
	TRAIN_DATA_RATIO = 1

	DETECTION_MAX_INSTANCES = 20

	SAVE_RESULT = True
```

- `inference_config.SAVE_RESULT = False` 

  infetence 결과를 `plt.show()` 를 통해 보여준다.

- `inference_config.SAVE_RESULT = True`

   infetence 결과에 대해 original input image와 비교하여 png file로 저장한다.

  

```python
inference_config = InferenceConfig()
# inference_config.display()
```



### set path directory

```python
# path of model to load weights
model_dir = os.path.join(os.getcwd(), "model_mask-rcnn")
# path of image to inference
path_dataset = os.path.join(os.getcwd() , 'test_image')

if inference_config.SAVE_RESULT:
	path_result = os.path.join(os.getcwd() , 'result_inference')
	os.makedirs(path_result, exist_ok=True)
```



### Create model in inference mode

```python
model = MaskRCNN(mode="inference", 
						config=inference_config,
						model_dir=model_dir)
```



### load weights

```python
model_path = model.find_last()
model.load_weights(model_path, by_name=True)
```





### detect and visualize

```python
for iter, path_ in enumerate(sorted(glob.glob (path_dataset + '\*.*'))):	
	title = path_.split("\\")[-1]
	original_image = cv2.imread(path_)	# <class 'numpy.ndarray'>
	original_image = utils.preprocessing_HE(original_image)
	print(f"file name : {title}")


	# results = ["rois" : [num_rois, (y1, x1, y2, x2)], 
	#			 "class_ids" : [num_rois]
	#			 "scores": [num_rois]
	#  			 "masks": [H, W, num_rois] 
	results = model.detect([original_image], verbose=1)


	r = results[0]
	visualize.display_instances(original_image, 
								r['rois'], r['masks'], r['class_ids'], r['scores'], 
								title, 
								save = inference_config.SAVE_RESULT, path = path_result)
```



- **`inference_config.SAVE_RESULT = True`** 인 경우

  image 저장

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/r1.png?raw=true)



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



# Create model in inference mode
model = MaskRCNN(mode="inference", 
						config=inference_config,
						model_dir=model_dir)
# load weights
model_path = model.find_last()
model.load_weights(model_path, by_name=True)

        
for iter, path_ in enumerate(sorted(glob.glob (path_dataset + '\*.*'))):	
	title = path_.split("\\")[-1]
	original_image = cv2.imread(path_)	# <class 'numpy.ndarray'>
	original_image = utils.preprocessing_HE(original_image)
	print(f"file name : {title}")


	# results = ["rois" : [num_rois, (y1, x1, y2, x2)], 
	#			 "class_ids" : [num_rois]
	#			 "scores": [num_rois]
	#  			 "masks": [H, W, num_rois] 
	results = model.detect([original_image], verbose=1)


	r = results[0]
	visualize.display_instances(original_image, 
								r['rois'], r['masks'], r['class_ids'], r['scores'], 
								title, 
								save = inference_config.SAVE_RESULT, path = path_result)
```

