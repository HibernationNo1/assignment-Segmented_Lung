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