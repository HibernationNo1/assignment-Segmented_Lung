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
#			learning_rate=trainig_config.LEARNING_RATE, 
#			epochs= 5, 
#			layers='heads') 

model.train(dataset_train, dataset_validation, 
 			learning_rate=trainig_config.LEARNING_RATE / 5,
 			epochs= 10, 
 			layers="all")
print("training 완료!!")

