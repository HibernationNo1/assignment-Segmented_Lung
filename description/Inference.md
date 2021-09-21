# Inference

### import

```python
import os

import visualize
import utils
import config 
from model import load_image_gt, MaskRCNN
```



### set Interence Config

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

  이미지 

```python
inference_config = InferenceConfig()
# inference_config.display()
```



### set path directory

```python
# path of model to load weights
model_dir = os.path.join(os.getcwd(), "mask-rcnn")
# path of dataset to load
path_dataset = os.path.join(os.getcwd(), 'test_dataset'  + '\dataset.json')

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



### load dataset

```python
dataset_test, _ = utils.load_dataset(inference_config.TRAIN_DATA_RATIO, path_dataset)
```



### detect and visualize

```python
for i in range(len(dataset_test)):
	image_id = i
	data_image = dataset_test[image_id]
	
    # convert to numpy and resize image for inference
	original_image, _ ,  _, _, _ = load_image_gt(data_image, image_id, inference_config)
	
    ## detect
	# results = ["rois" : [num_rois, (y1, x1, y2, x2)], 
	#			 "class_ids" : [num_rois]
	#			 "scores": [num_rois]
	#  			 "masks": [H, W, num_rois] 
	results = model.detect([original_image], verbose=1)


	r = results[0]
	title = data_image["image_info"]["file_name"]
	class_names = list()
	for annotation in dataset_test[image_id]["annotation"]:
		class_names.append(annotation["class_name"])
	# class_names = ["background", "left_lung", "right_lung"]
	
    ## visualize
	visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
								class_names, r['scores'], title, 
								save = inference_config.SAVE_RESULT, path = path_result)
```



- **`inference_config.SAVE_RESULT = True`** 인 경우

  image 저장

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/7.png?raw=true)



## Full code

```python
import os

import visualize
import utils
import config 
from model import load_image_gt, MaskRCNN

inference_config = config.InferenceConfig()
# inference_config.display()


## set path directory
# path of model to load weights
model_dir = os.path.join(os.getcwd(), "mask-rcnn")
# path of dataset to load
path_dataset = os.path.join(os.getcwd(), 'test_dataset'  + '\dataset.json')

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

dataset_test, _ = utils.load_dataset(inference_config.TRAIN_DATA_RATIO, path_dataset)

for i in range(len(dataset_test)):
	image_id = i
	data_image = dataset_test[image_id]

	original_image, _ ,  _, _, _ = load_image_gt(data_image, image_id, inference_config)

	# results = ["rois" : [num_rois, (y1, x1, y2, x2)], 
	#			 "class_ids" : [num_rois]
	#			 "scores": [num_rois]
	#  			 "masks": [H, W, num_rois] 
	results = model.detect([original_image], verbose=1)


	r = results[0]
	title = data_image["image_info"]["file_name"]
	class_names = list()
	for annotation in dataset_test[image_id]["annotation"]:
		class_names.append(annotation["class_name"])
	# class_names = ["background", "left_lung", "right_lung"]

	visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
								class_names, r['scores'], title, 
								save = inference_config.SAVE_RESULT, path = path_result)
```

