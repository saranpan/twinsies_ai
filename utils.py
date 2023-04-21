import torch
import streamlit as st
from streamlit_option_menu import option_menu
import torchvision.transforms as T
from PIL import Image
import random
import os

def pipeline(img,crop_model,emb_model,face_prob = 0.8, use_save_path=False):
    # Pipeline = crop_model + emb model 
    img_to_tensor = T.ToTensor()
    crop_img, face_prob = crop_model(img, return_prob=True ,save_path = 'temp_folder/crop_img.jpg')
    if use_save_path:
        crop_img = Image.open('temp_folder/crop_img.jpg')
        crop_img = img_to_tensor(crop_img)

    if face_prob is None:
        emb_img = None
    else:
        emb_img = emb_model(crop_img.unsqueeze(0))

    return emb_img, face_prob


def siamese_network(img_1,img_2,crop_model,emb_model,use_save_path = True):
    img_to_tensor = T.ToTensor()
    
    crop_img_1, prob_1 = crop_model(img_1,return_prob=True ,save_path = 'temp_folder/crop_img1.jpg')
    crop_img_2, prob_2 = crop_model(img_2,return_prob=True ,save_path = 'temp_folder/crop_img2.jpg')

    if use_save_path:
        crop_img_1 = Image.open('temp_folder/crop_img1.jpg')
        crop_img_2 = Image.open('temp_folder/crop_img2.jpg')

        crop_img_1 = img_to_tensor(crop_img_1)
        crop_img_2 = img_to_tensor(crop_img_2)

    # If prob is low, mean the cropped image is not confidence enough that its face
    emb_img_1 = emb_model(crop_img_1.unsqueeze(0))
    emb_img_2 = emb_model(crop_img_2.unsqueeze(0))
    
    l2_norm = compute_l2_dist(emb_img_1, emb_img_2)
    return l2_norm


def compute_l2_dist(img_1,img_2):
    """img 1, img_2 as tensor vector"""
    
    diff_emb = img_1 - img_2
    dist = torch.linalg.vector_norm(diff_emb)

    return dist

def get_random_file(directory ,blacklist_file : list = ['.DS_Store']):
    lst = []
    for file_ in os.listdir(directory):
        if file_ in blacklist_file:
            continue
        lst.append(file_)
    
    try:
        chosen_file = random.choice(lst)
    except IndexError:
        raise ValueError('This directory does not contain any files, except file in blacklist')

    return directory + '/' +chosen_file
