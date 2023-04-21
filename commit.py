# Commit to recreate image_tensor.pkl 

# import required module
import torch
import torchvision.transforms as T
from facenet_pytorch import MTCNN # For face alignment (crop + position)
from facenet_pytorch import InceptionResnetV1 # For embedding face to vector
import streamlit as st
from utils import compute_l2_dist, siamese_network, pipeline
from PIL import Image
import pickle
import io
import os

mtcnn = MTCNN(image_size=160, margin=0) # face align
resnet = InceptionResnetV1(pretrained='vggface2',classify=False).eval()

# assign directory
directory = 'database'
face_dct = {}

use_save_path = False
# iterate over files in
# that directory
for sub_dir in os.listdir(directory):
    if sub_dir == '.DS_Store':
        continue

    path = f'{directory}/{sub_dir}'
    face_lst = []
    for img in os.listdir(path): 
        if img == '.DS_Store':
            continue
        
        img = Image.open(f'{path}/{img}').convert('RGB')
        emb_vector, _ = pipeline(img,mtcnn,resnet, use_save_path)  
        face_lst.append(emb_vector)

    face_lst = torch.cat(face_lst)
    face_dct.update({sub_dir : face_lst})
    print('done preprocessing '+ sub_dir)

with open('emb_database/image_tensor.pkl', 'wb') as fp:
    pickle.dump(face_dct, fp)
    print('dictionary saved successfully to file')