# READ ME

서울대학병원 융합의학과 김영곤 교수님의 과제 수행 repository입니다.

reference : **[younggon2](https://github.com/younggon2)/[Research-Segmentation-Lung-CXR-COVID19](https://github.com/younggon2/Research-Segmentation-Lung-CXR-COVID19)**

dataset : [Covid Patients Chest X-Ray](https://www.kaggle.com/ankitachoudhury01/covid-patients-chest-xray)





## separation algorithm 

해당 project에서 다루는 data는 흉부의 CT사진입니다.

흉부의 CT사진을 찍는 과정에서,  대상자는 기기를 바라본 방향으로 흉부를 기기에 접촉시켜야 한다는 통일된 절차가 있습니다. 이는 곧 data의 형태가 일관적이라는 것을 의미합니다.

그렇기에 모든 CT image의 좌, 우는 통일되어 있으며, 이는 곧 object의 coordinate의 다양성이 크지 않다는 것이라 생각하였습니다.

그래서 object의 coordinate를 기준으로 left, right lung을 분할하여 구분하도록 code를 구성했습니다. 



#### 1. Find max, min coordinate of each object

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/3.png?raw=true)

좌표를 통해 lung을 구분하기 위해 먼저 image를 위에서 아래로 slicing하며 각 y좌표에 대한 object의 x좌표를 구한 후, 가장 큰 x좌표와 가장 작은 x좌표를 남겼습니다.

이미 object는 scipy.ndimage package의 label method에 의해 각각 고유한 number를 가지고 있기 때문에 각각의 object에 의한 x좌표의 최대, 최소가 도출됩니다.

```python
img_resize = image_resize(img_, IMAGE_SIZE)

labeled_array , feature_num = label(ensemble_mask_post_HF_EI)

for object_num in range(1, feature_num+1):
	min_index, max_index = IMAGE_SIZE[1], 0
	for i in range(int(IMAGE_SIZE[0])):  	# slicing			
		idx_temp = np.where(labeled_array[i] == object_num)	

		if np.shape(idx_temp)[1] !=0:
			max_index = max(max_index, idx_temp[0][-1])
			min_index = min(min_index, idx_temp[0][0])		
```



#### 2. Distinguish left, right

left lung, right lung의 기준을 각 object의 max, min coordinate가 속한 range에 따라 구별하도록 했습니다.

- left lung

  object의 x coordinate의 max value가 image width의 3/2를 넘어가지 않으며 x coordinate의 min value가 image width의 3/1을 넘어가지 않는 경우

- right lung

  object의 x coordinate의 max value가 image width의 3/2를 넘어가고 x coordinate의 min value가 image width의 3/1을 넘어가는 경우

  ![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/4.png?raw=true)

  ```python
  if max_index < int(IMAGE_SIZE[1]*2/3) and min_index < int(IMAGE_SIZE[1]*1/3):   # left lung
  					image_left_lung = get_lung_color_image(labeled_array, object_num, 'B')
  
  elif max_index > int(IMAGE_SIZE[1]*2/3) and min_index > int(IMAGE_SIZE[1]*1/3):	# right lung
  					image_right_lung = get_lung_color_image(labeled_array, object_num, 'R')
  
  else:
  					print("Separation failed!")
  
  color_mask_lung = image_left_lung + image_right_lung
  ```



##### get_lung_color_image

구분된 lung에 color를 입히는 함수입니다.

각 구분된 lung에 대해서 색으로 확인할 수 있도록 시각화 하기 위해 정의했습니다.

left lung의 color는 red, right lung의 color는 blue입니다.

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/2.png?raw=true)



```python
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
```



#### 3. Result 

각 image에 대해서 비교할 수 있도록 code를 구성했습니다.

![](https://github.com/HibernationNo1/assignment-Segmented_Lung/blob/master/image/result_0.png?raw=true)

##### function

```python
def Separation_l_r_lung(ensemble_mask_post_HF_EI, img_, path_, iter, path_save_comp_img):
	image_left_lung = np.zeros_like(IMAGE_SIZE).astype('uint8')
	image_right_lung = np.zeros_like(IMAGE_SIZE).astype('uint8')
	
	# image resize for addWeighted
	img_resize = image_resize(img_, IMAGE_SIZE)

	labeled_array , feature_num = label(ensemble_mask_post_HF_EI)

	for object_num in range(1, feature_num+1):
		min_index, max_index = IMAGE_SIZE[1], 0

		for i in range(int(IMAGE_SIZE[0])):  					# image의 위에서 아래로 slicing
			idx_temp = np.where(labeled_array[i] == object_num)	
			
			if np.shape(idx_temp)[1] !=0:
				max_index = max(max_index, idx_temp[0][-1])
				min_index = min(min_index, idx_temp[0][0])		

			if i == int(IMAGE_SIZE[0])-1:
				if max_index < int(IMAGE_SIZE[1]*2/3) and min_index < int(IMAGE_SIZE[1]*1/3):   # left lung
					image_left_lung = get_lung_color_image(labeled_array, object_num, 'B')

				elif max_index > int(IMAGE_SIZE[1]*2/3) and min_index > int(IMAGE_SIZE[1]*1/3):	# right lung
					image_right_lung = get_lung_color_image(labeled_array, object_num, 'R')

				else:
					print("Separation failed!")

	color_mask_lung = image_left_lung + image_right_lung

	# adding images by applying transparency
	alpha_img_resize = 0.8		# original image 투명도 
	beta_color_mask_lung = 1-alpha_img_resize
	res_image_lung = cv2.addWeighted(img_resize, alpha_img_resize, color_mask_lung, beta_color_mask_lung, 0)	

	fig, ax = plt.subplots(2, 3, figsize=(12, 12))

	fig.suptitle(str(path_.split('\\')[-1]), fontsize=25, fontweight = 'bold')
	fig.tight_layout()

	ax[0, 1].imshow(img_)
	ax[0, 1].set_title('Input with HE', fontsize = 20)

	ax[1, 0].imshow(ensemble_mask_post_HF_EI, cmap='gray')
	ax[1, 0].set_title('Ensemble + HF + EI', fontsize = 20)

	ax[1, 1].imshow(color_mask_lung)
	ax[1, 1].set_title('Color Ensemble + HF + EI ', fontsize = 20)

	ax[1, 2].imshow(res_image_lung)
	ax[1, 2].set_title('Color + Input Image', fontsize = 20)

	for axis in ax.flat:	# 좌표, 축 삭제(image만 보기 위해)
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)
		for _, spine in axis.spines.items():
			spine.set_visible(False)

	plt.savefig(path_save_comp_img + '/result_' + str(iter) +  '.png')
	# plt.show()
	plt.close()
```



