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

from PIL import Image

import sys
sys.path.append('/home/lcur1734/fromage') ##change this to your folder where you put the whole project

from fromage import models
from fromage import utils
from PIL import Image, UnidentifiedImageError


def retrieve_story(VIST_data:list):
    stories = []
    for story in VIST_data:
        story_valid = check_story(story)
        if None not in story_valid[0]:
            stories.append(story_valid) 
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
            data -- story sequences from VIST (5 images + 5 captions for each story sequence)
            caption -- how many previous captions to input
            image -- how many previous images to input
            recall -- represents k in recall

    Return: generated images and correponding story id
    """

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
        prompt = [item for pair in zip_longest(prompt_images, prompt_captions, fillvalue=None) for item in pair if item is not None]
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
        if 'url_o' in item[1].keys():
            img = utils.get_image_from_url(item[1]['url_o'])
        else:
            img = utils.get_image_from_url(item[1]['url_m'])
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
            data -- story sequences from VIST (5 images + 5 captions for each story sequence)
            caption -- how many previous captions to input
            image -- how many previous images to input
            recall -- represents k in recall

    Return: generated images and correponding story id
    """
    outputs_images, output_targets, output_ids = generate_output(model, VIST_data, caption, image, recall)
    ## Create path for the first time
    ##if not os.path.exists(save_path):
        ##os.makedirs(save_path)
    ## Save results in npz file
    outputs_images = [[np.asarray(y) for y in x[1]] for x in outputs_images]
    output_targets = [np.asarray(x) for x in output_targets]
    with open(f'{save_path}/EX1_R{recall}_C{caption}_I{image}.npz', 'wb') as f:
        np.savez(f, images_output=outputs_images, images_target = output_targets, story_ids=output_ids)


def __main__():
    # ### Load Model and Embedding Matrix
    # Load model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)

    print("-- Loading data:")
    # Load VIST dataset for experiment
    file_path = 'VIST_expriment.json'
    with open(file_path, 'r') as f:
        VIST_data = json.load(f)
    random.seed(42)
    VIST_data  = random.sample(VIST_data,1000)
    print(f"-- Finish loading | {len(VIST_data)} stories")

    ## Define path to save results
    save_path = "/home/lcur1747/fromage/results"

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
