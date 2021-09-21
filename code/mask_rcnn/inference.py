import os

import visualize
import utils
import config 
from model import load_image_gt, MaskRCNN

inference_config = config.InferenceConfig()
# inference_config.display()


## set path directory
# path of model to load weights
model_dir = os.path.join(os.getcwd(), "model_mask-rcnn")
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