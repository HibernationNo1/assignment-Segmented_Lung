import os

import config 
from model import MaskRCNN
import utils


### set path dir
# path of model to save 
model_dir = os.path.join(os.getcwd(), "model_mask-rcnn")
os.makedirs(model_dir, exist_ok=True) 

# path of dataset to load
path_dataset = os.path.join(os.getcwd(), 'training_dataset'  + '\dataset.json')


trainig_config = config.TrainConfig()
# trainig_config.display()

#load dataset
dataset_train, dataset_validation = utils.load_dataset(trainig_config.TRAIN_DATA_RATIO, path_dataset)


### training
model = MaskRCNN(mode="training", config = trainig_config, model_dir = model_dir)


#model.train(dataset_train, dataset_validation, 
# 			learning_rate=trainig_config.LEARNING_RATE / 5,
# 			epochs= 15, 
# 			layers="all")


model.load_weights(model.find_last(), by_name=True)
model.train(dataset_train, dataset_validation, 
			learning_rate=trainig_config.LEARNING_RATE, 
			epochs= 3, 
			layers='heads') 

print("training 완료!!")

