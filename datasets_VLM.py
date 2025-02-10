import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
# import cv2
import os
from PIL import Image
import pandas as pd
import pdb
import os
import json
import random
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

random.seed(1234)
np.random.seed(1234)


def gen_dataset(region_fp, tokenizer):

    catg_name = []
    catg_mask = []
    labels = torch.zeros(len(region_fp), dtype=torch.long)

    cnt_reg_withpoi = 0

    for r in tqdm(region_fp):
        reg = json.load(open(f"./datasets/source/{r}"))
        n = reg["poi_names"]
        if len(n) != 0:
            cnt_reg_withpoi += 1
    
        random.shuffle(n)
        name = " ".join(n)

        encode_dict = tokenizer(name, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        tokens = encode_dict['input_ids']
        masks = encode_dict['attention_mask']
        catg_name.append(tokens)
        catg_mask.append(masks)

        labels[len(catg_name)-1] = reg["labels"][0]

    print(f"cnt_reg_withpoi: {cnt_reg_withpoi}")

    return catg_name, catg_mask, labels, region_fp

class my_Dataset(Dataset):
    def __init__(self, phase, cities = ["beijing"], pre_process=None,n_class=4):
        
        self.n_class = n_class
        self.pre_process = pre_process

        self.poi_tokenizer = BertTokenizer.from_pretrained('../bert-cache/')
        self.phase = phase
        folder = "./datasets/source/"
        files = os.listdir(folder)
        ## 
        files = [f for f in files if f.endswith(".json")]
        print(len(files))

        files = [f for f in files if f.split("_")[0] in cities]

        random.seed(1234)
        np.random.seed(1234)
 
        train_files = random.sample(files, int(len(files) * 0.8))

        test_files = [f for f in files if f not in train_files]
        random.shuffle(test_files)
        print(len(train_files), len(test_files))

        if phase == "train":
            self.names, self.masks, self.labels, self.r_id = \
                gen_dataset(train_files, self.poi_tokenizer)
            
        elif phase == "test":
            self.names, self.masks, self.labels, self.r_id =\
                 gen_dataset(test_files, self.poi_tokenizer)
        
        print(f"len of {phase} dataset: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rid = self.r_id[idx]
        n = self.names[idx]
        n_mask = self.masks[idx]

        if self.phase in ["train","test"]:
            label = self.labels[idx]
        else:
            label = -1

        img1 = Image.open(f"./datasets/source/{rid.split('.')[0]}.png").convert('RGB')
        
        if os.path.exists(f"./datasets/source/{rid.split('.')[0]}_height.png"):
            img2 = Image.open(f"./datasets/source/{rid.split('.')[0]}_height.png").convert('RGB')
        else:
            img2 = np.zeros((224,224,3))
            img2 = Image.fromarray(img2.astype('uint8')).convert('RGB')

        if os.path.exists(f"./datasets/source/{rid.split('.')[0]}_building.png"):
            img3 = Image.open(f"./datasets/source/{rid.split('.')[0]}_building.png").convert('RGB')
        else:
            print(f"Warning: {rid.split('.')[0]}_building.png not found")
            img3 = np.zeros((224,224,3))
            img3 = Image.fromarray(img3.astype('uint8')).convert('RGB')

        img1 = self.pre_process(img1)
        img2 = self.pre_process(img2)
        img3 = self.pre_process(img3)
        return rid, n, n_mask, img1, label, img2, img3

    def get_distribution(self):
        dist = torch.zeros(self.n_class)
        for l in self.labels:
            dist[l] += 1

        return dist