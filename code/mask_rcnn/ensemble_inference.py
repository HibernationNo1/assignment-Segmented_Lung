import os
import cv2
import glob

import ensemble_visualize
import utils
import config 
from model import MaskRCNN

inference_config = config.InferenceConfig()
# inference_config.display()D


## set path directory
# path of model to load weights
model_dir = os.path.join(os.getcwd(), "model_mask-rcnn")
# path of image to inference
path_dataset = os.path.join(os.getcwd() , 'sample_test_image')

if inference_config.SAVE_RESULT:
	path_result = os.path.join(os.getcwd() , 'result_inference')
	os.makedirs(path_result, exist_ok=True)  


model_list = list()
for i in range(1, 4):
	model = MaskRCNN(mode="inference", 
						config=inference_config,
						model_dir=model_dir)
	# load weights
	model_path = model.find_last(file_name = "_" + str(i))
	model.load_weights(model_path, by_name=True)
	model_list.append(model)

        
for iter, path_ in enumerate(sorted(glob.glob (path_dataset + '\*.*'))):	
	title = path_.split("\\")[-1]
	original_image = cv2.imread(path_)	# <class 'numpy.ndarray'>
	original_image = utils.preprocessing_HE(original_image)

	# original_image = original_image*0.9	더 안좋아짐
	print(f"file name : {title}")

	result_list = list()
	for i in range(3):
		results = model_list[i].detect([original_image], verbose=1)
		result_list.append(results)
	# results = ["rois" : [num_rois, (y1, x1, y2, x2)], 
	#			 "class_ids" : [num_rois]
	#			 "scores": [num_rois]
	#  			 "masks": [H, W, num_rois] 


	ensemble_visualize.display_instances(original_image, 
								result_list, 
								title, 
								save = inference_config.SAVE_RESULT, path = path_result)