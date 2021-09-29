import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import os


import colorsys
from skimage.measure import find_contours
from matplotlib.patches import Polygon

def ensemble_mask(masks_list):
	if len(masks_list) == 0:
		return None 

	ensemble_mask = np.zeros(shape = masks_list[0].shape)
	for mask in masks_list:
		ensemble_mask += mask
	ensemble_mask[ensemble_mask<=0.0] = 0
	ensemble_mask[ensemble_mask> 0.0] = 1
	return ensemble_mask

def ensemble_bbox(image, bbox_list): 
	if len(bbox_list) == 0:
		return None 

	y_min = image.shape[1]
	y_max = 0
	x_min = image.shape[0]
	x_max = 0
	for bbox in bbox_list:
		y1, x1, y2, x2 = bbox
		y_min = min(y1, y_min)
		x_min = min(x1, x_min)
		y_max = max(y2, y_max)
		x_max = max(x2, x_max)
	return [y_min, x_min, y_max, x_max]

def ensemble_score(scores_list):
	if len(scores_list) == 0:
		return None 

	average_score = 0
	for scores in scores_list:
		average_score += scores
	
	average_score /= len(scores_list)
	return average_score

def ensemble_data(class_ids, class_ids_list, image, N,
				  boxes_list, masks_list, scores_list):
	left_lung_bbox_list = list()
	right_lung_bbox_list = list()
	left_lung_masks_list = list()
	right_lung_masks_list = list()
	left_lung_score_list = list()
	right_lung_score_list = list()
	for i in range(3):

		if boxes_list[i].shape[0] > 0 : 

			for j in range(boxes_list[i].shape[0]):
				for class_id in class_ids:
					if class_id == class_ids_list[i][j]:

						if class_id == 1:
							left_lung_bbox_list.append(boxes_list[i][j])
							left_lung_masks_list.append(masks_list[i][j])
							left_lung_score_list.append(scores_list[i][j])
						elif class_id == 2:
							right_lung_bbox_list.append(boxes_list[i][j])
							right_lung_masks_list.append(masks_list[i][j])
							right_lung_score_list.append(scores_list[i][j])

	boxes = list()
	masks = list()
	scores = list()
	for id in class_ids:
		if id == 1:
			boxes.append(ensemble_bbox(image, left_lung_bbox_list))
			masks.append(ensemble_mask(left_lung_masks_list))
			scores.append(ensemble_score(left_lung_score_list))
		elif id == 2:
			boxes.append(ensemble_bbox(image, right_lung_bbox_list))
			masks.append(ensemble_mask(right_lung_masks_list))
			scores.append(ensemble_score(right_lung_score_list))

	return boxes, masks, scores


def random_colors(N, bright=True):
	"""
	Generate random colors.
	To get visually distinct colors, generate them in HSV space then
	convert to RGB.
	"""
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors

def apply_mask(image, mask, color, alpha=0.5):
	"""Apply the given mask to the image.
	"""
	for c in range(3):
		image[:, :, c] = np.where(mask == 1, 
						 image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
						 image[:, :, c])
	return image


def display_instances(image, result_list,
					  title = " ", figsize = (15, 15), save = False, path = None):

	boxes_list = list()
	masks_list = list()
	class_ids_list = list()
	scores_list = list()
	for i in range(3):
		result = result_list[i]
		r = result[0]

		# r['rois'] = [(y1, x1, y2, x2), (y1, x1, y2, x2), ...] 
		boxes_list.append(r['rois'])

		# r['masks'].shape =  [w, h, num_mask]
		mask = list()
		for j in range(r['masks'].shape[2]):
			mask.append(r['masks'][:, :, j])
		masks_list.append(mask)

		# r['class_ids'] = [id, id, ...]
		class_ids_list.append(r['class_ids'])

		# r['scores'] = [score, score, ...]
		scores_list.append(r['scores'])



	class_names = ["background", "left lung", "right lung"]
	N = max((boxes_list[0].shape[0], boxes_list[1].shape[0], boxes_list[2].shape[0]))

	if not N:
		print("*** No instances to display ***")
	else:
		for i in range(3):
			assert boxes_list[i].shape[0] == len(masks_list[i]) == class_ids_list[i].shape[0]
	
	if not save : 
		figsize = (7, 7)
	fig, ax = plt.subplots(1, 2, figsize=figsize)

	colors = random_colors(N)

		# Show area outside image boundaries.
	height, width = image.shape[:2]

	fig.suptitle(title, fontsize=25, fontweight = 'bold')
	fig.tight_layout()
	ax[0].imshow(np.array(image, dtype = np.uint8))
	ax[0].set_ylim(height + 10, -10)
	ax[0].set_xlim(-10, width + 10)
	ax[0].axis('off')
	ax[0].set_title('Orininal image', fontsize = 30)
	ax[1].set_ylim(height + 10, -10)
	ax[1].set_xlim(-10, width + 10)
	ax[1].axis('off')
	ax[1].set_title('Result image', fontsize = 30)

	if height < 512 : 
		thickness = 1
		y_text_loc = 2
	elif 512 <= height and  height < 1024 : 
		thickness = 2
		y_text_loc = 4
	elif 1024 <= height :
		thickness = 3
		y_text_loc = 7

	# class_ids : 가장 많은 instance를 detection한 model의 r['class_ids']
	class_ids = list()
	for i in range(3):
		if N == boxes_list[i].shape[0]:
			class_ids = class_ids_list[i]

	boxes, masks, scores = ensemble_data(class_ids, class_ids_list, image, N,
				  						 boxes_list, masks_list, scores_list)

	# 받아온 image는 scalar임
	masked_image = np.array(image, dtype = np.uint8).copy()
	for i in range(N):
		if boxes[i] == None:
			continue

		# Bbox
		y1, x1, y2, x2 = boxes[i]

		# color
		color = colors[i]
		color_cv2 = tuple(int(i)*255 for i in color)
		
		cv2.rectangle(masked_image, (x1, y1), (x2, y2), color_cv2, thickness)
		
		# put class name, score
		class_id = class_ids[i]
		score = scores[i]
		label = class_names[class_id]

		text = label + " " + str(f"{score:.3f}")
		bottomLeftCornerOfText = (x1, y1-y_text_loc)
		font = cv2.FONT_HERSHEY_SIMPLEX
		fontScale = height/1000
		fontColor = color_cv2
		lineType = cv2.LINE_AA
		
		cv2.putText(masked_image, text,
					bottomLeftCornerOfText,
					font,
					fontScale,
					fontColor,
					thickness = thickness,
					lineType = lineType)

		# apply mask
		mask = masks[i]
		masked_image = apply_mask(masked_image, mask, color)


		# Mask Polygon
		padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		contours = find_contours(padded_mask, 0.5)
		for verts in contours:
			# Subtract the padding and flip (y, x) to (x, y)
			verts = np.fliplr(verts) - 1
			p = Polygon(verts, facecolor="none", edgecolor=color)
			ax[1].add_patch(p)
	

	ax[1].imshow(masked_image.astype(np.uint8))

	if save and os.path.exists(path):
		plt.savefig(path + '/' + title.split('.')[0] +  '.png')
		print(f"{title.split('.')[0]} saved successfully. \n")
	elif not save:
		plt.show()
	plt.close()