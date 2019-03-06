# -*- coding: utf-8 -*-
import os, scipy.misc
from glob import glob
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms
import json



class Data:
    def __init__(self,folder,transform,shuffle=True):
        self.datapath = folder
        if not os.path.isdir(folder):
            print("the folder does not exist")
        self.files = os.listdir(self.datapath)
        self.transform = transform
        self.shuffle=shuffle
        if shuffle:
            np.random.shuffle(self.files)    
        self.count = 0

    def next(self):
        if self.count+1 >= len(self.files):
            if self.shuffle:
                np.random.shuffle(self.files)
            self.count = 0
        imgs = []
        filename = self.files[self.count]
        file_path = os.path.join(self.datapath,filename)
        img = Image.open(file_path)

        result_img = self.transform(img)
        result_img = result_img.unsqueeze(0)
                
        self.count = self.count + 1
        return result_img, filename

    def get(self,filename):
        file_path = os.path.join(self.datapath,filename)
        if not os.path.isfile(file_path):
            return None
        img = Image.open(file_path)
        result_img = self.transform(img)
        result_img = result_img.unsqueeze(0)
        return result_img