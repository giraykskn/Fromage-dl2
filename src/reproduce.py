#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy
import torch
import random
import json
from transformers import logging
import itertools
from itertools import zip_longest
import os
logging.set_verbosity_error()
import logging
import time
import pickle
import argparse
from PIL import Image
import sys
sys.path.append('/home/lcur1748/fromage') ##change this to your folder where you put the whole project
from fromage import models
from fromage import utils
from PIL import Image, UnidentifiedImageError
# Create a logger
logger = logging.getLogger('my_logger')
# Set the logging level (optional)
logger.setLevel(logging.DEBUG)
# create a file handler
# Set the format for log message
file_handler = logging.FileHandler("reproduce.log")
logger.addHandler(file_handler)   
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

def retrieve_story(VIST_data:list):
    stories = []
    for story in VIST_data:
        story_valid = check_story(story)
        if story_valid != None:
            stories.append(story_valid) 
        else:
            print("invalid story") 
    return stories


## FUNCTION OF MODEL RETRIEVING IMAGES
def generate_output(model, stories:list, caption:int=1, image:int=0, recall:int=1):
    """
    This function reproduces experiments for the following settings:
    1. inputs with 1 caption 
    2. inputs with 5 captions
    3. inputs with 5 captions and 4 images

    Inputs:
            model -- FROMAGE model
            storiees -- story sequences from VIST (5 images + 5 captions for each story sequence)
            caption -- how many previous captions to input
            image -- how many previous images to input
            recall -- represents k in recall

    Return: generated images and correponding story id
    """
    logger.info("Caption: {}, Image : {}, Recall : {}".format(caption,image,recall))
    prompts = []
    targets = []
    story_ids = []
    for story in stories:
        ## get images and captions needed
        prompt_images = story[0][:image]
        target_image = story[0][-1]
        prompt_captions = story[1][-caption:]
        prompt_captions[-1] = f"{prompt_captions[-1]}[RET]"
        story_id = story[2]
        prompt = [item for pair in zip_longest(prompt_captions, prompt_images, fillvalue=None) for item in pair if item is not None]
        ## add it to the prompt lists
        targets.append(target_image)
        story_ids.append(story_id)
        prompts.append(prompt)
    ## Inferecing
    outputs_images = []
    output_targets = []
    output_ids = []
    for i, prompt in enumerate(prompts):
        output = model.generate_for_images_and_texts(prompt, max_img_per_ret=recall)
        if len(output[1]) == recall:
            outputs_images.append(output)
            output_targets.append(targets[i])
            output_ids.append(story_ids[i])
        else:
            print(f"Story {story_ids[i]} has invalid images")
    return outputs_images, output_targets, output_ids 


def check_story(story:list):
    images = []
    captions = []
    story_ids = None
    for item in story:
        try:
            if 'url_o' in item[1].keys():
                img = utils.get_image_from_url(item[1]['url_o'])
            else:
                img = utils.get_image_from_url(item[1]['url_m'])
        except:
            return None
        caption = item[0]['original_text']
        images.append(img)
        captions.append(caption)
        story_ids = item[0]['story_id']
    return images, captions, story_ids



## MAIN FUCNTION TO RUN EXPERIMENTS AND STORE OUTPUTS
def run_experiment(model, save_path:str, VIST_data:list, caption:int=1, image:int=0, recall:int=1):
    """
    This function reproduces experiments for the following settings:
    1. inputs with 1 caption 
    2. inputs with 5 captions
    3. inputs with 5 captions and 4 images

    Inputs:
            model -- FROMAGE model
            save_path -- path to save results
            VIST_data -- story sequences from VIST (5 images + 5 captions for each story sequence)
            caption -- how many previous captions to input
            image -- how many previous images to input
            recall -- represents k in recall

    Return: generated images and correponding story id
    """
    outputs_images, output_targets, output_ids = generate_output(model, VIST_data, caption, image, recall)
    ## Create path for the first time
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ## Save results in npz file
    outputs_images = [[np.asarray(y) for y in x[1]] for x in outputs_images]
    output_targets = [np.asarray(x) for x in output_targets]


    recall_ = 0
    # For each target image
    for index,target_image in enumerate(output_targets):
        # Check the predicted images for this
        for predicted_image in outputs_images[index]:
            # If we have predicted_image == target_image then add one to the recall
            if np.array_equal(target_image, predicted_image):
                recall_ += 1
                break
            

    logger.debug(f"Length of output images: {len(outputs_images)}, shape of the first image: {outputs_images[0][0].shape}")
    logger.debug(f"Length of the output targets: {len(output_targets)}, shape of the first target: {output_targets[0].shape}")
    logger.debug(f"Recall at @{recall} is {recall_}")
    with open(f'{save_path}/EX1_R{recall}_C{caption}_I{image}.npz', 'wb') as f:
        np.savez(f, images_output=outputs_images, images_target = output_targets, story_ids=output_ids)

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
    random.seed(42)
    # VIST_data  = random.sample(VIST_data,10)
    print(f"-- Finish loading | {len(VIST_data)} stories")

    ## Define path to save results
    save_path = "/Results"

    ## retrieve all stories 
    print("Retrieving stories ...")
    stories = retrieve_story(VIST_data)
    print(f"Retrieval finished. {len(stories)} stories retrieved.")

    ## Define experiment configurations
    configs = [(1,0,1), (1,0,5), (1,0,10), (5,0,1),(5,0,5),(5,0,10),(5,4,1),(5,4,5),(5,4,10)] #e.g. (1,0,1) represents inputs with 1 caption no image at recall@1
    for config in configs:
        print(f"--- Experiment ongoing - caption{config[0]} / image{config[1]} / recall{config[2]}...")
        run_experiment(model, save_path, stories, config[0], config[1], config[2])
        print(f"--- Experiment finished")


__main__()
