#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy
import torch
import random
import json
from transformers import logging
import itertools
logging.set_verbosity_error()

from PIL import Image

import sys
sys.path.append('/home/lcur1679/fromage/final/src') ##change this to your folder where you put the whole project

from fromage import models
from fromage import utils
from PIL import Image, UnidentifiedImageError


## FUNCTION OF MODEL RETRIEVING IMAGES
def generate_output(model, VIST_data:list, caption:int=1, image:bool=False, recall:int=1):
    """
    This function reproduces experiments for the following settings:
    1. inputs with 1 caption 
    2. inputs with 5 captions
    3. inputs with 5 captions and 4 images

    Inputs:
            model -- FROMAGE model
            data -- story sequences from VIST (5 images + 5 captions for each story sequence)
            caption -- how many previous captions to input
            image -- how many previous images to input
            recall -- represents k in recall

    Return: generated images and correponding story id
    """

    ## Flatten data to a 1d list
    data = [i for story in VIST_data for i in story]
    ## 1 caption
    if caption == 1:
        prompts = [[f"{data[i][0]['original_text']}[RET]"] for i in range(4,len(data),5)] #size=(num_story*1)
        story_ids = [data[i][0]['story_id'] for i in range(4,len(data),5)] #size=(num_story*1)
    ## 5 captions
    elif caption == 5 and not image:
        prompts = [[data[j][0]['original_text'] if j!=i+5-1 else f"{data[j][0]['original_text']}[RET]" for j in range(i, i+caption)] for i in range(0, len(data), caption)] #size=(num_story*5)        
        story_ids = [data[i][0]['story_id'] for i in range(4,len(data),5)] #size=(num_story*1)
    ## 5 captions and 4 images
    else:
        prompts = []
        story_ids = [] #final size = (num_story*9)
        for i in range(0, len(data), 5):
            story_prompt = []
            for j in range(i, i+5):
                if j!=i+4:
                    if 'url_o' in data[j][1].keys():
                        story_prompt.append(utils.get_image_from_url(data[j][1]['url_o']))
                    else:
                        story_prompt.append(utils.get_image_from_url(data[j][1]['url_m']))
                    story_prompt.append(data[j][0]['original_text'])
                else:
                    story_prompt.append(data[j][0]['original_text'])                  
            if len(story_prompt)==9 and None not in story_prompt:
                prompts.append(story_prompt)
                story_ids.append(data[i][0]['story_id'])
    
    ## Inferecing
    model_outputs = [] #size=(num_story*recall)
    for i, prompt in enumerate(prompts):
        output = model.generate_for_images_and_texts(prompt, max_img_per_ret=recall)
        if len(output[1]) == recall:
            model_outputs.append(output)
        else:
            print(f"Not enough images")
            story_ids[i] = None
    return model_outputs, story_ids 



## MAIN FUCNTION TO RUN EXPERIMENTS AND STORE OUTPUTS
def run_experiment(model, save_path:str, VIST_data:list, caption:int=1, image:bool=False, recall:int=1):
    """
    This function reproduces experiments for the following settings:
    1. inputs with 1 caption 
    2. inputs with 5 captions
    3. inputs with 5 captions and 4 images

    Inputs:
            model -- FROMAGE model
            save_path -- path to save results
            data -- story sequences from VIST (5 images + 5 captions for each story sequence)
            caption -- how many previous captions to input
            image -- how many previous images to input
            recall -- represents k in recall

    Return: generated images and correponding story id
    """
    model_outputs, story_ids = generate_output(model, VIST_data, caption, image, recall)
    ## Create path for the first time
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ## Save results in npz file
    with open(f'{save_path}/EX1_R{recall}_C{caption}.npz', 'wb') as f:
        np.savez(f, images=model_outputs, story_ids=story_ids)


def __main__():
    # ### Load Model and Embedding Matrix
    # Load model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)

    print("-- Loading data:")
    # Load VIST dataset for experiment
    file_path = 'VIST_data_for_experiments.json'
    with open(file_path, 'r') as f:
        VIST_data = json.load(f)
    print(f"-- Finish loading | {len(VIST_data)} stories")

    ## Define path to save results
    save_path = "\results"

    ## Define experiment configurations
    caption_image = [(1,False),(5,False),(5,True)]
    recall = [1,5,10]
    config_combinations = list(itertools.product(caption_image,recall))

    for config in config_combinations:
        if config[0][1]==False:
            print(f"--- Experiment ongoing - caption{config[0][0]} without images...")
            run_experiment(model, save_path, VIST_data, config[0][0], config[0][1], config[1])
            print(f"--- Experiment finished")
        else:
            print(f"--- Experiment ongoing - caption{config[0][0]} with 4 images...")
            run_experiment(model, save_path, VIST_data, config[0][0], config[0][1], config[1])
            print(f"--- Experiment finished")               

__main__()
