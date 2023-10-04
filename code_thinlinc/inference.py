import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.graves import graves_aug as graves


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#GRAVES_MODEL_PATH = "/work1/thogujo/logs/graves20191205T1211/mask_rcnn_graves_0032.h5"

GRAVES_MODEL_PATH = "/work3/thgujo/logs/mask_rcnn_graves_0032.h5"

#%%
class InferenceConfig(graves.CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.9
    
inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = GRAVES_MODEL_PATH

assert model_path != ""
print("Loading weights from ", model_path)
model.load_weights(model_path,by_name=True)

#%%
dataset = graves.CustomDataset()
dataset.load_custom("/zhome/2e/9/124284/environments/gravestones_env/Code/Mask_RCNN/samples/graves/dataset_g/","val")
#dataset.load_custom("/media/thorsteinngj/c1f0e49d-1218-4d11-ab9f-aadb4a021648/home/thorsteinngj/Documents/Thesis/pics/")

# Must call before using the dataset
dataset.prepare()

#%%
import pandas as pd
from time import time




#%%
import skimage
import re

test_dir = '/work3/thgujo/Pictures/1944'
image_paths = []
for filename in  os.listdir(test_dir):
    if os.path.splitext(filename)[1].lower() in ['.jpg','.png','.jpeg']:
        image_paths.append(os.path.join(test_dir,filename))
        
image_paths = image_paths
datapoints = []
res = []
t0 = time()
for i in tqdm(range(len(image_paths))):
    try:
        img = skimage.io.imread(image_paths[i])
        if len(np.shape(img)) > 2 and (np.shape(img)[2] == 3):
            img_arr = np.array(img)
            results = model.detect([img_arr],verbose=1)
            r = results[0]
            #print(r['class_ids'])
            
            zeros = 0
            ones = 0
            twos = 0
            threes = 0 
            fours = 0
            fives = 0
            sixes = 0
            sevens = 0
            if r['class_ids'].size == 0:
                zeros += 1
            else:
                for j in range(len(r['class_ids'])):
                    if r['class_ids'][j] == 1:
                        ones +=1
                    elif r['class_ids'][j] == 2:
                        twos +=1
                    elif r['class_ids'][j] == 3:
                        threes +=1  
                    elif r['class_ids'][j] == 4:
                        fours +=1
                    elif r['class_ids'][j] == 5:
                        fives +=1
                    elif r['class_ids'][j] == 6:
                        sixes +=1
                    else:
                        sevens += 1
                    
                    
            #Þarf að læra að save-a myndir eftir þvi hvort þær seu png, jpeg, jpg, gif ...
            datapoint = re.findall('\d+',image_paths[i])[-1]
            print(i)
            print(datapoint)
            
            #Getting the values
            t1 = time()
            tot_time = t1-t0
            
            data = {'grave_id': int(datapoint),'no_sign':zeros,'angels':ones,"bibles":twos,"crosses":threes,"david_star":fours,"doves":fives,"persons":sixes,"praying_hands":sevens,"time_elapsed":tot_time}
        	
            datapoints.append(data)
    except:
        print("Couldn't process image")





my_dataset_df = pd.DataFrame.from_dict(datapoints)
my_dataset_df.to_csv('/work3/thgujo/dataframes/1944.csv')
