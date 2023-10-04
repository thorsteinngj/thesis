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

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/thorsteinngj/Documents/Skoli/Thesis/Code/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.graves import graves_new as graves


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
GRAVES_MODEL_PATH = "/media/thorsteinngj/c1f0e49d-1218-4d11-ab9f-aadb4a021648/home/thorsteinngj/Documents/Thesis/logs/graves20191205T1211/mask_rcnn_graves_0032.h5"  # TODO: update this path
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
dataset.load_custom("/home/thorsteinngj/Documents/Skoli/Thesis/Code/Mask_RCNN/samples/graves/dataset_g/","val")
#dataset.load_custom("/media/thorsteinngj/c1f0e49d-1218-4d11-ab9f-aadb4a021648/home/thorsteinngj/Documents/Thesis/pics/")
#dataset.load_custom(BALLOON_DIR, "/dataset/val")

# Must call before using the dataset
dataset.prepare()

#%%
import pandas as pd
from time import time




#%%
import skimage
import re
from tqdm import tqdm

test_dir = '/media/thorsteinngj/c1f0e49d-1218-4d11-ab9f-aadb4a021648/home/thorsteinngj/Documents/Thesis/pics/'
image_paths = []
for filename in  os.listdir(test_dir):
    if os.path.splitext(filename)[1].lower() in ['.jpg','.png','.jpeg','.gif']:
        image_paths.append(os.path.join(test_dir,filename))

datapoints = []
res = []
t0 = time()
i=0
for image_path in tqdm(range(len(image_paths))):    
    try:
        img = skimage.io.imread(image_paths[image_path])
        if len(np.shape(img)) > 2 and (np.shape(img)[2] == 3):
                img_arr = np.array(img)
                results = model.detect([img_arr],verbose=1)
                r = results[0]
                print(r['class_ids'])
                
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
                    for i in range(len(r['class_ids'])):
                        if r['class_ids'][i] == 1:
                            ones +=1
                        elif r['class_ids'][i] == 2:
                            twos +=1
                        elif r['class_ids'][i] == 3:
                            threes +=1  
                        elif r['class_ids'][i] == 4:
                            fours +=1
                        elif r['class_ids'][i] == 5:
                            fives +=1
                        elif r['class_ids'][i] == 6:
                            sixes +=1
                        else:
                            sevens += 1
                        
                        
                #Þarf að læra að save-a myndir eftir þvi hvort þær seu png, jpeg, jpg, gif ...
                datapoint = re.findall('\d+',image_paths[image_path])[-1]
                
                #Getting the values
                t1 = time()
                tot_time = t1-t0
                
                data = {'grave_id': int(datapoint),'no_sign':zeros,'angels':ones,"bibles":twos,"crosses":threes,"david_star":fours,"doves":fives,"persons":sixes,"praying_hands":sevens,"time_elapsed":tot_time}
            	
                datapoints.append(data)       
    except:
        print("Error")

#%%
my_dataset_df = pd.DataFrame.from_dict(datapoints)
my_dataset_df.to_csv('/media/thorsteinngj/c1f0e49d-1218-4d11-ab9f-aadb4a021648/home/thorsteinngj/Documents/Thesis/pics/test.csv')
    
#%%

import skimage
import re
from tqdm import tqdm

test_dir = '/media/thorsteinngj/c1f0e49d-1218-4d11-ab9f-aadb4a021648/home/thorsteinngj/Documents/Thesis/pics/'
image_paths = []
for filename in  os.listdir(test_dir):
    if os.path.splitext(filename)[1].lower() in ['.jpg','.png','.jpeg','.gif']:
        image_paths.append(os.path.join(test_dir,filename))

img = image_paths[36]
img = skimage.io.imread(img)
img_arr = np.array(img)
results = model.detect([img_arr],verbose=1)
r = results[0]
visualize.display_instances(img, r['rois'],r['masks'],r['class_ids'],dataset.class_names,r['scores'],figsize=(5,5))
