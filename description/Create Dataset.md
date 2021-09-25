# Create Dataset

해당 code는 input image로부터 lungs의 mask image를 얻기 위해 서울대학병원 융합의학과 김영곤 교수님의 [Research-Segmentation-Lung](https://github.com/younggon2/Research-Segmentation-Lung-CXR-COVID19)를 인용했습니다.

- mask image로부터 letf, right lungs를 구분했습니다.
- segmentation을 수행하는 Mask R-CNN의 학습을 위해 image의 meta data를 json file로 저장하도록 구현했습니다.



## Find max, min coordinate of each object

해당 project에서 다루는 data는 흉부의 CT사진입니다.

흉부의 CT사진을 찍는 과정에서,  대상자는 기기를 바라본 방향으로 흉부를 기기에 접촉시켜야 한다는 통일된 절차가 있습니다. 이는 곧 data의 형태가 일관적이라는 것을 의미합니다.

그렇기에 모든 CT image의 좌, 우는 통일되어 있으며, 이는 곧 object의 coordinate의 다양성이 크지 않다는 것이라 생각하였습니다.

그래서 object의 coordinate를 기준으로 left, right lung을 분할하여 구분하도록 code를 구성했습니다. 



#### Find x coordinate

좌표를 통해 lung을 구분하기 위해 먼저 image를 위에서 아래로 slicing하며 각 y좌표에 대한 object의 x좌표를 구한 후, 가장 큰 x좌표와 가장 작은 x좌표를 남겼습니다.

이미 object는 scipy.ndimage package의 label method에 의해 각각 고유한 number를 가지고 있기 때문에 각각의 object에 의한 x좌표의 최대, 최소가 도출됩니다.

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/3.png?raw=true)



### Distinguish left, right

left lung, right lung의 기준을 각 object의 x max, min coordinate가 속한 range에 따라 구별하도록 했습니다.

- left lung

  object의 x coordinate의 max value가 image width의 3/2를 넘어가지 않으며 x coordinate의 min value가 image width의 3/1을 넘어가지 않는 경우

- right lung

  object의 x coordinate의 max value가 image width의 3/2를 넘어가고 x coordinate의 min value가 image width의 3/1을 넘어가는 경우
  
  ![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/0.png?raw=true)

위의 기준으로 left, right구분을 하여 color를 통해 표현해보면 아래와 같습니다.

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/2.png?raw=true)



#### Find y coordinate

x좌표를 구하는 방식과 동일합니다.

image를 `y = -x + height` 직선을 기준으로 대칭이동 한 후 object의 y좌표를 구하여 가장 큰 y좌표와 가장 작은 y좌표를 남기는 방식으로 object에 의한 y좌표의 최대, 최소가 도출되도록 했습니다.



x min, max좌표와 y min, max좌표를 구함으로 bounding box의 top-left, bottom-right의 좌표를 알 수 있으며

이를 통해 image의 meta data를 구성하고 json file로 저장합니다.



## Meta data

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/1.png?raw=true)

input image에 대해서 image informaion, original image, instance information에 대한 값을 계산했습니다.

instance information은 background, left lung, right lung 세 가지의 instance에 대한 data를 얻도록 했습니다.

**instance information**

- class_id 
  - background : 0
  - left lung : 1
  - right lung : 2
- class_name
  - background : "background"
  - left lung : ''left lung"
  - right lung : "right lung"
- bbox : 각 instance에 대한 top left, bottom right 좌표입니다.
- height, width : 각 instance에 대한 height, width입니다.
- mask_image : 각 instance에 대한 mask image에서 각 pixel의 value를 True, False의 boolean으로 구성한 image입니다.



```python
def meta_data_image(resized_img, mask_img_l, mask_img_r, iter, 
				bbox_l, bbox_r, background_mask):

	instance_info = [
		{
			"class_id": 0,
			"class_name" : "background",
			"bbox" : [0, 0, IMAGE_SIZE[0], IMAGE_SIZE[1] ], # y_min_l, x_min_l, y_max_l, x_max_l
			"height" : IMAGE_SIZE[0],
			"width" : IMAGE_SIZE[1],
			"mask_image" : background_mask
		},

		{
			"class_id": 1,
			"class_name" : "left_lung",
			"bbox" : bbox_l, # y_min_l, x_min_l, y_max_l, x_max_l
			"height" : bbox_l[2] - bbox_l[0],
			"width" : bbox_l[3] - bbox_l[1],
			"mask_image" : mask_img_l
		},

		{
			"class_id": 2,
			"class_name" : "right_lung",
			"bbox" : bbox_r, # y_min_r, x_min_r, y_max_r, x_max_r
			"height" : bbox_r[2] - bbox_r[0],
			"width" : bbox_r[3] - bbox_r[1],
			"mask_image" : mask_img_r		
		}
	]

	image_info = {
			"image_id" : iter
			"width" : IMAGE_SIZE[1],
			"height" :	IMAGE_SIZE[0],
			"file_name" : str(iter) + ".jpg"
	}

	image = {
		"original_image" : resized_img
	}

	data_image = {
			"annotation" : annotation,
			"image" : image,
			"image_info" : image_info
			
	}

	return data_image
```





> dataset을 json형식으로 save할 때 numpy의 dtype에 대해 읽지 못하는 issue가 있었습니다.
>
> ```
> TypeError: Object of type int64 is not JSON serializable
> ```
>
> 이를 해결하기 위해 `NpEncoder` class를 선언했으며, Mask R-CNN에서 dataset을 load하여 활용할 때
>
> np.array()를 통해 다시 numpy의 type으로 변경해주어야 하는 과정이 필요합니다.
>
> ```python
> class NpEncoder(json.JSONEncoder):
>     def default(self, obj):
>         if isinstance(obj, np.integer):
>             return int(obj)
>         elif isinstance(obj, np.floating):
>             return float(obj)
>         elif isinstance(obj, np.ndarray):
>             return obj.tolist()
>         else:
>             return super(NpEncoder, self).default(obj)
> ```
>





## Result

input image에 대해서 lungs의 mask image와 left, right구분에 관한 결과입니다.

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/4.png?raw=true)

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/6.png?raw=true)

## Full Code

```python
import segmentation_models as sm
import glob
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage
from scipy.ndimage import label

from absl import app

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# change framework of sm from keras to tensorflow.keras
sm.set_framework('tf.keras')

IMAGE_SIZE = (256, 256,3)	

# Parameter
path_base_model = os.path.join(os.getcwd() , 'code' + '\create_dataset' + '\models')
path_base_input = os.path.join(os.getcwd() , 'code' + '\create_dataset' + '\\test_input_dataset')  

path_base_result = os.path.join(os.getcwd() , 'code' + '\create_dataset' + '\\result_tmp')
os.makedirs(path_base_result, exist_ok=True)  
path_save_training_dataset = os.path.join(os.getcwd() , 'test_dataset') # instance for save of distinguish image 
os.makedirs(path_save_training_dataset, exist_ok=True)



# Model loads
BACKBONE = 'efficientnetb0'
model1 = sm.Unet(BACKBONE, input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),classes=1, activation='sigmoid',encoder_weights='imagenet')
model2 = sm.Unet(BACKBONE, input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),classes=1, activation='sigmoid',encoder_weights='imagenet')
model3 = sm.Unet(BACKBONE, input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),classes=1, activation='sigmoid',encoder_weights='imagenet')

BACKBONE = 'efficientnetb7'
model4 = sm.Unet(BACKBONE, input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),classes=1, activation='sigmoid',encoder_weights='imagenet')
model5 = sm.Unet(BACKBONE, input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),classes=1, activation='sigmoid',encoder_weights='imagenet')

preprocess_input = sm.get_preprocessing(BACKBONE)

# load pre-trained model weights 
model1.load_weights(path_base_model + '\model1.hdf5')
model2.load_weights(path_base_model + '\model2.hdf5')
model3.load_weights(path_base_model + '\model3.hdf5')
model4.load_weights(path_base_model + '\model4.hdf5')
model5.load_weights(path_base_model + '\model5.hdf5')


class NpEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.bool):
			return bool(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(NpEncoder, self).default(obj)

# Histogram Equalization
def preprocessing_HE(img_):
    hist, _ = np.histogram(img_.flatten(), 256,[0,256])		# histogram
    cdf = hist.cumsum()										# 누적합
    cdf_m = np.ma.masked_equal(cdf,0)						# 0인 element는 mask처리
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min()) # Histogram equalization
    cdf = np.ma.filled(cdf_m,0).astype('uint8')				# np.ma.masked_equal로 인해 mask처리된 element를 0으로 
    img_2 = cdf[img_]										# original image에 historam적용
    
    return img_2  
        
def get_binary_mask(mask_, th_ = 0.5):
    mask_[mask_>th_]  = 1
    mask_[mask_<=th_] = 0
    return mask_
    
def ensemble_results(mask1_, mask2_, mask3_, mask4_, mask5_):
    # predicted mask image의 element가 0.5보다 높으면 1, 같거나 낮으면 0
    mask1_ = get_binary_mask(mask1_)
    mask2_ = get_binary_mask(mask2_)
    mask3_ = get_binary_mask(mask3_)
    mask4_ = get_binary_mask(mask4_)
    mask5_ = get_binary_mask(mask5_)
    
	# 모든 model의 predicted mask image를 합산 후 조건
    ensemble_mask = mask1_ + mask2_ + mask3_ + mask4_ + mask5_
    ensemble_mask[ensemble_mask<=2.0] = 0
    ensemble_mask[ensemble_mask> 2.0] = 1
    
    return ensemble_mask

def postprocessing_HoleFilling(mask_):
	ensemble_mask_post_temp = ndimage.binary_fill_holes(mask_).astype(int)
	return ensemble_mask_post_temp

def get_maximum_index(labeled_array):
	ind_nums = []
	# objct가 2개라고 할 때 np.unique(labeled_array) == 0, 1, 2
	# ind_num = [[0, 0], [0, 1], [0, 2]]
	for i in range (len(np.unique(labeled_array)) - 1):
		ind_nums.append ([0, i+1])

	# ind_num = [[1번 object인 pixel의 개수, 0], [2번 object인 pixel의 개수, 1], ...
	for i in range (1, len(np.unique(labeled_array))):
		ind_nums[i-1][0] = len(np.where(labeled_array == np.unique(labeled_array)[i])[0])

	ind_nums = sorted(ind_nums)
	
	# pixel의 개수가 가장 많은 object 2개의 number 반환
	return ind_nums[len(ind_nums)-1][1], ind_nums[len(ind_nums)-2][1]
    

def postprocessing_EliminatingIsolation(ensemble_mask_post_temp):
    # 각 obejct에 대해 number를 부여하여 image의 object의 각 pixel마다 number를 곱한다.
	labeled_array, _ = label(ensemble_mask_post_temp)

	# lung으로 가장 유력해 보이는 object의 number 반환
	ind_max1, ind_max2 = get_maximum_index(labeled_array)
    
	# 가장 유력한 object 2개만 따로 추려내서 mask image로 그린다.
	ensemble_mask_post_temp2 = np.zeros(ensemble_mask_post_temp.shape)
	ensemble_mask_post_temp2[labeled_array == ind_max1] = 1
	ensemble_mask_post_temp2[labeled_array == ind_max2] = 1    

	return ensemble_mask_post_temp2.astype(int)


def image_resize(img_, IMAGE_SIZE):
	if np.shape(img_)[0] > IMAGE_SIZE[0] and np.shape(img_)[1] > IMAGE_SIZE[1] : # original image가 기대하는 input image보다 작을 때
		# image를 더 큰 size로 resize(Bilinear Interpolation 사용)
		img_resize = cv2.resize(img_,(IMAGE_SIZE[0],IMAGE_SIZE[1]),cv2.INTER_LINEAR)

	else : 	# original image가 기대하는 input image보다 클 때
		# image를 더 작은 size로 resize(Area Interpolation 사용)
		img_resize = cv2.resize(img_,(IMAGE_SIZE[0],IMAGE_SIZE[1]),cv2.INTER_AREA)
	
	return img_resize


def get_prediction(model_, img_, IMAGE_SIZE):  # 수정
	img_resize = image_resize(img_, IMAGE_SIZE)

	img_org_resize_HE = preprocessing_HE(img_resize)    	# Histogram Equalization
	img_ready = preprocess_input(img_org_resize_HE)			# backbone에 알맞게 전처리
	img_ready = np.expand_dims(img_ready, axis=0) 			# (256, 256, 3) → (1, 256, 256, 3)
	pr_mask = model_.predict(img_ready)			# input data에 대해 model이 학습한 parameters를 기반으로 predict
	pr_mask = np.squeeze(pr_mask, axis = 0)
	return pr_mask[:,:,0]


def get_lung_color_image(labeled_array, object_num, color):
	labeled_array_temp = np.zeros(labeled_array.shape)
	labeled_array_temp[labeled_array == object_num] = 1

	if color == 'R':
		lung_B = labeled_array_temp * 0
		lung_G = labeled_array_temp * 0
		lung_R = labeled_array_temp * 255
	elif color == 'B':
		lung_B = labeled_array_temp * 255
		lung_G = labeled_array_temp * 0
		lung_R = labeled_array_temp * 0

	image_lung = np.dstack([lung_B, lung_G, lung_R]).astype('uint8')

	return image_lung


def find_coordinate(resized_img, ensemble_mask_post_HF_EI):

	# 전체 scale에 0.9를 곱해서 1인 pixel은 0.9로 만든다. >> lung 좌표 찾을 때 1과 명확한 판별을 하기 위해
	tmp_resized_img = np.zeros(resized_img.shape).astype('uint8')
	alpha_img_resize = 0.9		# original image 투명도 
	beta_color_mask_lung = 1-alpha_img_resize
	resized_img_f_d = cv2.addWeighted(resized_img, alpha_img_resize, tmp_resized_img, beta_color_mask_lung, 0)

	labeled_array , feature_num = label(ensemble_mask_post_HF_EI)

	x_min_l, x_max_l, x_min_r, x_max_r = 0, 0, 0, 0
	y_min_l, y_max_l, y_min_r, y_max_r = 0, 0, 0, 0
	for object_num in range(1, feature_num+1):
		x_min_coord, x_max_coord = IMAGE_SIZE[1], 0
		tmp_segmentation_list = list()
		tmp_area = 0
		for i in range(int(IMAGE_SIZE[0])):  # y축 slicing
			idx_temp = np.where(labeled_array[i] == object_num)	
			
			if np.shape(idx_temp)[1] !=0:
				x_max_coord = max(x_max_coord, idx_temp[0][-1])
				x_min_coord = min(x_min_coord, idx_temp[0][0])
				for x_coordinate in range(idx_temp[0][0], idx_temp[0][-1] + 1):
					tmp_segmentation_list.extend([x_coordinate, i])	
					tmp_area +=1

			if i == int(IMAGE_SIZE[0])-1:
				labeled_array_temp = np.zeros(labeled_array.shape)
				# object의 가장 오른쪽 pixel 좌표가 image의 2/3 지점 좌표보다 작고,
				# object의 가장 왼쪽 pixel 좌표가 image의 1/3 지점 좌표보다 작으면 left lung
				if x_max_coord < int(IMAGE_SIZE[1]*2/3) and x_min_coord < int(IMAGE_SIZE[1]*1/3):   # left lung
					x_min_l, x_max_l = x_min_coord, x_max_coord
					
					labeled_array_temp[labeled_array == object_num] = 1
					label_image_left_lung = resized_img_f_d.copy()	
					# gray scale의 mask 영역 image (256, 256, 3)
					label_image_left_lung[labeled_array_temp != 1] = 0
					
					# 0, 1로만 구성된 mask image  (256, 256, 1)
					mask_image_left_lung = np.expand_dims(labeled_array_temp.astype(bool), axis = 2)
					image_left_lung = get_lung_color_image(labeled_array, object_num, 'B')

				elif x_max_coord > int(IMAGE_SIZE[1]*2/3) and x_min_coord > int(IMAGE_SIZE[1]*1/3):	# right lung
					x_min_r, x_max_r = x_min_coord, x_max_coord
					
					labeled_array_temp[labeled_array == object_num] = 1
					label_image_right_lung = resized_img_f_d.copy()
					label_image_right_lung[labeled_array_temp != 1] = 0

					mask_image_right_lung = np.expand_dims(labeled_array_temp.astype(bool), axis = 2)
					image_right_lung = get_lung_color_image(labeled_array, object_num, 'R')
				else:
					print("Separation failed!")
		
		if object_num == feature_num:
			# labeled_array을 사용해서 slicing을 하면 x축 slicing이 되지 않는다.
			# labeled_array을 대각선(y = -x + 256) 대칭
			dig_symt_labeled_array = np.zeros(labeled_array.shape)
			for m in range(np.shape(labeled_array)[0]):
				for n in range(np.shape(labeled_array)[1]):
					dig_symt_labeled_array[m][n] = labeled_array[n][m]
			
			image_height = np.shape(dig_symt_labeled_array)[0]
			for object_num_tmp in range(1, feature_num+1):
				y_min_coord, y_max_coord = IMAGE_SIZE[0], 0
				
				check_tmp = 0  
				# labeled_array의 object number가 1 or 2로 random하게 주어지기 때문에
				# left, right 구별간에 0~(x_max_l + x_min_r)/2 구간에서 object가 detection되었는지 check하기 위함
				for k in range(int(image_height)):
					idx_temp = np.where(dig_symt_labeled_array[k] == object_num_tmp)
					if np.shape(idx_temp)[1] !=0:
						y_max_coord = max(y_max_coord, idx_temp[0][-1])
						y_min_coord = min(y_min_coord, idx_temp[0][0])	
						check_tmp = 1							

					if k == int((x_max_l + x_min_r)/2):   	# left lung
						if check_tmp == 1:
							y_min_l, y_max_l = y_min_coord, y_max_coord	
							y_min_coord, y_max_coord = IMAGE_SIZE[0], 0
							break
						elif check_tmp == 0:
							continue
					
					if k == image_height - 1:				# right lung
						if check_tmp == 1:
							y_min_r, y_max_r = y_min_coord, y_max_coord
	
	x_center_l, y_center_l = int((x_min_l + x_max_l)/2), int((y_min_l + y_max_l)/2)
	x_center_r, y_center_r = int((x_min_r + x_max_r)/2), int((y_min_r + y_max_r)/2)
							
	Region_img = label_image_left_lung + label_image_right_lung
	color_mask_lung = image_left_lung + image_right_lung

	mask_image = mask_image_left_lung + mask_image_right_lung
	background_mask = np.ones(shape = mask_image.shape)
	background_mask[mask_image == 1] = 0

	bbox_l = [y_min_l, x_min_l, y_max_l, x_max_l]
	coordi_l = [y_center_l, x_center_l]
	bbox_r = [y_min_r, x_min_r, y_max_r, x_max_r]
	coordi_r = [y_center_r, x_center_r]

	return (Region_img, mask_image_left_lung, mask_image_right_lung, background_mask,
			color_mask_lung, 
			bbox_l, coordi_l,
			bbox_r, coordi_r)



def draw_GT_box(resized_img, bbox_l, coordi_l, bbox_r, coordi_r):
	GT_image = resized_img.copy()

	y_min_l, x_min_l, y_max_l, x_max_l = bbox_l
	y_center_l, x_center_l = coordi_l

	y_min_r, x_min_r, y_max_r, x_max_r = bbox_r
	y_center_r, x_center_r = coordi_r

	pt1_l, pt2_l = (x_min_l, y_max_l), (x_max_l, y_min_l)
	cv2.rectangle(GT_image, pt1_l, pt2_l, color = (255, 0, 0), thickness = None, lineType = None, shift =None)
	cv2.circle(GT_image, (x_center_l, y_center_l), radius = 3, color = (255, 0, 0), thickness = -1)

	pt1_r, pt2_r = (x_min_r, y_max_r), (x_max_r, y_min_r)
	cv2.rectangle(GT_image, pt1_r, pt2_r, color = (0, 0, 255), thickness = None, lineType = None, shift =None)
	cv2.circle(GT_image, (x_center_r, y_center_r), radius = 3, color = (0, 0, 255), thickness = -1)

	return GT_image

def meta_data_image(resized_img, mask_img_l, mask_img_r, iter, 
				bbox_l, bbox_r, background_mask):

	instance_info = [
		{
			"class_id": 0,
			"class_name" : "background",
			"bbox" : [0, 0, IMAGE_SIZE[0], IMAGE_SIZE[1] ], # y_min_l, x_min_l, y_max_l, x_max_l
			"height" : IMAGE_SIZE[0],
			"width" : IMAGE_SIZE[1],
			"mask_image" : background_mask
		},

		{
			"class_id": 1,
			"class_name" : "left lung",
			"bbox" : bbox_l, # y_min_l, x_min_l, y_max_l, x_max_l
			"height" : bbox_l[2] - bbox_l[0],
			"width" : bbox_l[3] - bbox_l[1],
			"mask_image" : mask_img_l
		},

		{
			"class_id": 2,
			"class_name" : "right lung",
			"bbox" : bbox_r, # y_min_r, x_min_r, y_max_r, x_max_r
			"height" : bbox_r[2] - bbox_r[0],
			"width" : bbox_r[3] - bbox_r[1],
			"mask_image" : mask_img_r		
		}
	]

	image_info = {
			"image_id" : iter, 
			"width" : IMAGE_SIZE[1],
			"height" :	IMAGE_SIZE[0],
			"file_name" : str(iter) + ".jpg"
	}

	image = {
		"original_image" : resized_img
	}

	data_image = {
			"annotation" : instance_info,
			"image" : image,
			"image_info" : image_info
			
	}

	return data_image


def save_fig_images(Region_img, resized_img, res_image_lung, original_img, path_, iter):
	fig, ax = plt.subplots(2, 3, figsize=(12, 12))
	ax[0, 1].imshow(original_img, cmap='gray')
	ax[0, 1].set_title('Orininal_image', fontsize = 20)

	ax[1, 0].imshow(Region_img)
	ax[1, 0].set_title('Mask_img', fontsize = 20)

	ax[1, 1].imshow(resized_img, cmap='gray')
	ax[1, 1].set_title('GT_Box_image', fontsize = 20)

	ax[1, 2].imshow(res_image_lung, cmap='gray')
	ax[1, 2].set_title('Projection_imge', fontsize = 20)

	for axis in ax.flat:
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)
		for spine_loc, spine in axis.spines.items():
			spine.set_visible(False)

	fig.suptitle(str(path_.split('\\')[-1]), fontsize=25, fontweight = 'bold')
	fig.tight_layout()
	plt.savefig(path_base_result + '/result_' + str(iter) +  '.png')
	plt.close()



def make_metadata_image(ensemble_mask_post_HF_EI, img_, path_, iter):
	# image resize for addWeighted
	resized_img = image_resize(img_, IMAGE_SIZE)
	original_img = resized_img.copy()

	
	# find GT_box coordinate
	# coordinate의 0, 0기준은 image의 left-top이다.
	(Region_img, mask_img_l, mask_img_r, background_mask, color_mask_lung, 
	bbox_l, coordi_l,
	bbox_r, coordi_r) = find_coordinate(resized_img, ensemble_mask_post_HF_EI)

	# adding images by applying transparency
	alpha_img_resize = 0.8		# original image 투명도 
	beta_color_mask_lung = 1-alpha_img_resize
	res_image_lung = cv2.addWeighted(resized_img, alpha_img_resize, color_mask_lung, beta_color_mask_lung, 0)	

	# GT_box가 표현된 image draw
	GT_image_img = draw_GT_box(resized_img, 
							   bbox_l, coordi_l,
							   bbox_r, coordi_r,)
	
	# mask, GT_box, projection image를 plt로 graw 후 저장
	save_fig_images(Region_img, GT_image_img, res_image_lung, original_img, path_, iter)

	# image로부터 뽑은 infomation으로 dataset을 만든다.
	data_image = meta_data_image(resized_img, mask_img_l, mask_img_r, iter,
				bbox_l, bbox_r, background_mask)

	return data_image


# inference
def main(_):
	# image가 저장된 directory에서 각 image를 가져온다.
	dataset_json = list()

	for iter, path_ in enumerate(sorted(glob.glob (path_base_input + '\*.*'))):		 
		file_name = path_.split("\\")[-1]
		print (f'file: {file_name}, iter : {iter}')
		
		img = cv2.imread(path_)    
		# input image에 대한 model의 predictor 반환
		pr_mask1 = get_prediction (model1, img, IMAGE_SIZE) 
		pr_mask2 = get_prediction (model2, img, IMAGE_SIZE)
		pr_mask3 = get_prediction (model3, img, IMAGE_SIZE)
		pr_mask4 = get_prediction (model4, img, IMAGE_SIZE)
		pr_mask5 = get_prediction (model5, img, IMAGE_SIZE)    

		# ensemble mask 계산	
		ensemble_mask            = ensemble_results(pr_mask1, pr_mask2, pr_mask3, pr_mask4, pr_mask5)
		
		# mask image의 각 object안의 빈 공간을 채워넣는다.
		ensemble_mask_post_HF    = postprocessing_HoleFilling(ensemble_mask)

		# draw Lung mask image
		ensemble_mask_post_HF_EI = postprocessing_EliminatingIsolation(ensemble_mask_post_HF)

		data_image = make_metadata_image(ensemble_mask_post_HF_EI, img, path_, iter)
		
		dataset_json.append(data_image)
	
	print("saving file...")

	# dataset_json =json.dumps(dataset_json, cls = NpEncoder)
	with open(path_save_training_dataset + '\dataset.json', 'w', encoding='utf-8') as make_file:
		json.dump(dataset_json, make_file, ensure_ascii=False, indent="\t", cls=NpEncoder)

	print("end")

if __name__ == '__main__':  
	app.run(main)
```

