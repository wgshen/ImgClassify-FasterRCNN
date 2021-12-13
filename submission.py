GOOGLE_DRIVE_PATH = './'

import numpy as np
import torch
from tqdm import trange
import torch.nn as nn
# More imports
import sys
sys.path.append(GOOGLE_DRIVE_PATH)

import time, os
# os.environ["TZ"] = "US/Eastern"
# time.tzset()

from two_stage_detector import hello_two_stage_detector
hello_two_stage_detector()

from a5_helper import hello_helper
hello_helper()

two_stage_detector_path = os.path.join(GOOGLE_DRIVE_PATH, 'two_stage_detector.py')
two_stage_detector_edit_time = time.ctime(os.path.getmtime(two_stage_detector_path))
print('two_stage_detector.py last edited on %s' % two_stage_detector_edit_time)

import eecs598
import torch
import torch.nn as nn
import torch.nn.functional as F

from a5_helper import *

if torch.cuda.is_available:
    print('Has GPU!')
    dev = 'cuda:0'
else:
    print('No GPU.')
    dev = 'cpu'

from two_stage_detector import TwoStageDetector

# Unfortunately, I accidentally deleted the network with 0.69365 score
# This is the best saved network
weights_path = os.path.join(GOOGLE_DRIVE_PATH, "frcnn_detector_0.67844.pt")
frcnn_detector = TwoStageDetector(num_classes=3).to(dtype=torch.float32, device=dev)
frcnn_detector.load_state_dict(torch.load(weights_path, map_location=torch.device(dev)))
frcnn_detector.eval()

# Load filenames for testing data
from glob import glob
from PIL import Image, ImageOps
files = glob('test/*/*_image.jpg')
files.sort()
test_filenames = []
for file in files:
    guid = file.split('\\')[-2]
    idx = file.split('\\')[-1].replace('_image.jpg', '')
    test_filenames.append(guid+'/'+idx)
n_test = len(test_filenames)

down_sampling = 1
new_height = int(1052 / down_sampling)
new_width = int(1914 / down_sampling)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __getitem__(self, idx):
        # load images and masks
        img_path = './test/' + self.filenames[idx] + '_image.jpg'
        img = Image.open(img_path).convert("RGB")
        img = img.resize((new_width, new_height))
        box = [0, 0, 0, 0, 0]
        
        info = {'annotation': {}}
        info['annotation']['filename'] = self.filenames[idx]
        info['annotation']['object'] = [{'name': str(box[-1]),
                                         'bndbox': {'xmin': str(box[0]), 
                                                    'ymin': str(box[1]),
                                                    'xmax': str(box[2]), 
                                                    'ymax': str(box[3])}}]
        return img, info

    def __len__(self):
        return len(self.filenames)
    
test_dataset = TestDataset(test_filenames)

device = torch.device(dev)
labels_test = []
toFiles = ['guid/image,label\n']
test_loader = pascal_voc2007_loader(test_dataset, batch_size=1)
test_loader = iter(test_loader)
count = 0
with torch.no_grad():
    for i in trange(n_test):
        img, box, _, _, filename = test_loader.next()
        proposals, conf_scores, classes = frcnn_detector.inference(img.to(device), thresh=0.5)
        try:
            label = classes[0][0].item()
        except:
            label = 1
            count += 1
        labels_test.append(label)
        toFiles.append(filename[0]+','+str(label)+'\n')

f = open('submission.txt', 'w')
f.writelines(toFiles)
f.close()