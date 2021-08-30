# READ ME

서울대학병원 융합의학과 김영곤 교수님의 과제 수행 repository입니다.

reference : **[younggon2](https://github.com/younggon2)/[Research-Segmentation-Lung-CXR-COVID19](https://github.com/younggon2/Research-Segmentation-Lung-CXR-COVID19)**

dataset : [Covid Patients Chest X-Ray](https://www.kaggle.com/ankitachoudhury01/covid-patients-chest-xray)





## create label data

해당 project에서 다루는 data는 흉부의 CT사진입니다.

흉부의 CT사진을 찍는 과정에서,  대상자는 기기를 바라본 방향으로 흉부를 기기에 접촉시켜야 한다는 통일된 절차가 있습니다. 이는 곧 data의 형태가 일관적이라는 것을 의미합니다.

그렇기에 모든 CT image의 좌, 우는 통일되어 있으며, 이는 곧 object의 coordinate의 다양성이 크지 않다는 것이라 생각하였습니다.

그래서 object의 coordinate를 기준으로 left, right lung을 분할하여 구분하도록 code를 구성했습니다. 



#### 1. Find max, min coordinate of each object

#### 1. Find x coordinate

좌표를 통해 lung을 구분하기 위해 먼저 image를 위에서 아래로 slicing하며 각 y좌표에 대한 object의 x좌표를 구한 후, 가장 큰 x좌표와 가장 작은 x좌표를 남겼습니다.

이미 object는 scipy.ndimage package의 label method에 의해 각각 고유한 number를 가지고 있기 때문에 각각의 object에 의한 x좌표의 최대, 최소가 도출됩니다.

##### 

##### Distinguish left, right

left lung, right lung의 기준을 각 object의 max, min coordinate가 속한 range에 따라 구별하도록 했습니다.

- left lung

  object의 x coordinate의 max value가 image width의 3/2를 넘어가지 않으며 x coordinate의 min value가 image width의 3/1을 넘어가지 않는 경우

- right lung

  object의 x coordinate의 max value가 image width의 3/2를 넘어가고 x coordinate의 min value가 image width의 3/1을 넘어가는 경우



#### 2. Find y coordinate

이번엔 x좌표에 대한 object의 y표를 구한 후, 가장 큰 y좌표와 가장 작은 y좌표를 남기는 방식으로 object에 의한 y좌표의 최대, 최소가 도출되도록 했습니다.

```python
def find_coordinate(resized_img, ensemble_mask_post_HF_EI):

	# 전체 scale에 0.9를 곱해서 1인 pixel은 0.9로 만든다. >> lung 좌표 찾을 때 1과 명확한 판별을 하기 위해
	tmp_resized_img = np.zeros(resized_img.shape).astype('uint8')
	alpha_img_resize = 0.9		# original image 투명도 
	beta_color_mask_lung = 1-alpha_img_resize
	resized_img_f_d = cv2.addWeighted(resized_img, alpha_img_resize, tmp_resized_img, beta_color_mask_lung, 0)

	labeled_array , feature_num = label(ensemble_mask_post_HF_EI)

	x_min_l, x_max_l, x_min_r, x_max_r = 0, 0, 0, 0
	y_min_l, y_max_l, y_min_r, y_max_r = 0, 0, 0, 0
	x_center_l, y_center_l, x_center_r, y_center_r = 0, 0, 0, 0
	for object_num in range(1, feature_num+1):
		x_min_coord, x_max_coord = IMAGE_SIZE[1], 0
		for i in range(int(IMAGE_SIZE[0])):  # y축 slicing
			idx_temp = np.where(labeled_array[i] == object_num)	
			
			if np.shape(idx_temp)[1] !=0:
				x_max_coord = max(x_max_coord, idx_temp[0][-1])
				x_min_coord = min(x_min_coord, idx_temp[0][0])	

			if i == int(IMAGE_SIZE[0])-1:
				labeled_array_temp = np.zeros(labeled_array.shape)
				# object의 가장 오른쪽 pixel 좌표가 image의 2/3 지점 좌표보다 작고,
				# object의 가장 왼쪽 pixel 좌표가 image의 1/3 지점 좌표보다 작으면 left lung
				if x_max_coord < int(IMAGE_SIZE[1]*2/3) and x_min_coord < int(IMAGE_SIZE[1]*1/3):   # left lung
					x_min_l, x_max_l = x_min_coord, x_max_coord
					
					labeled_array_temp[labeled_array == object_num] = 1
					label_image_left_lung = resized_img_f_d.copy()	
					label_image_left_lung[labeled_array_temp != 1] = 0
					
					image_left_lung = get_lung_color_image(labeled_array, object_num, 'B')
				elif x_max_coord > int(IMAGE_SIZE[1]*2/3) and x_min_coord > int(IMAGE_SIZE[1]*1/3):	# right lung
					x_min_r, x_max_r = x_min_coord, x_max_coord
					
					labeled_array_temp[labeled_array == object_num] = 1
					label_image_right_lung = resized_img_f_d.copy()
					label_image_right_lung[labeled_array_temp != 1] = 0
					
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
							
	mask_img = label_image_left_lung + label_image_right_lung
	color_mask_lung = image_left_lung + image_right_lung

	return (mask_img, color_mask_lung, 
			x_min_l, y_max_l, x_max_l, y_min_l, x_center_l, y_center_l,
			x_min_r, y_max_r, x_max_r, y_min_r, x_center_r, y_center_r)	
```





