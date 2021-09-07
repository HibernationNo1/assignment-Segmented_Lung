import os
import sys
import numpy as np


from config import TrainConfig
from model import MaskRCNN
import utils

# path of model to save 
model_dir = os.path.join(os.getcwd(), "mask-rcnn")
os.makedirs(model_dir, exist_ok=True) 

# path of dataset to load
# 여기 나중에 수정
path_dataset = os.path.join(os.getcwd(), 'result_dataset'  + '\dataset.json')

config = TrainConfig()
# config.display()

# load dataset
#dataset_train, dataset_validation = utils.load_dataset(config.TRAIN_DATA_RATIO, path_dataset)

model = MaskRCNN(config.MODE, config, model_dir)
print("됐다.")

# model.train(dataset_train, dataset_validation, 
#            learning_rate=config.LEARNING_RATE, 
#            epochs=1, 
 #           layers='heads')