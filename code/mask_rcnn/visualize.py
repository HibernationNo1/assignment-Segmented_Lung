import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import os


import colorsys
from skimage.measure import find_contours
from matplotlib.patches import Polygon


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


def display_instances(image, boxes, masks, class_ids, class_names, scores=None,
					  title = " ", figsize = (15, 15), save = False, path = None):
	N = boxes.shape[0]

	if not N:
		print("\n*** No instances to display *** \n")
	else:
		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
	
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



	# 받아온 image는 scalar임
	masked_image = np.array(image, dtype = np.uint8).copy()
	for i in range(N):
		# color
		color = colors[i]
		color_cv2 = tuple(int(i)*255 for i in color)

		# Bbox
		y1, x1, y2, x2 = boxes[i]
		cv2.rectangle(masked_image, (x1, y1), (x2, y2), color_cv2, 1)
		
		# put class name, score
		class_id = class_ids[i]
		score = scores[i]
		label = class_names[class_id]

		text = label + " " + str(f"{score:.3f}")
		bottomLeftCornerOfText = (x1, y2+7)
		font = cv2.FONT_HERSHEY_SIMPLEX
		fontScale = 0.3
		fontColor = color_cv2
		thickness = 1
		lineType = cv2.LINE_AA
		
		cv2.putText(masked_image, text,
					bottomLeftCornerOfText,
					font,
					fontScale,
					fontColor,
					thickness = thickness,
					lineType = lineType)

		# apply mask
		mask = masks[:, :, i]
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
	elif not save:
		plt.show()
	plt.close()