#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import copy
import torch
import random
import json
from transformers import logging
import itertools

logging.set_verbosity_error()

from PIL import Image

import sys

sys.path.append('/home/lcur1734/Fromage-dl2')  ##change this to your folder where you put the whole project

from fromage import models
from fromage import utils
from PIL import Image, UnidentifiedImageError


## FUNCTION OF MODEL RETRIEVING IMAGES
def generate_output(model, open_ended_data: list, shot: int = 1, recall: int = 1):
    """
    This function reproduces experiments for the following settings:
    1. inputs 1 shot
    2. inputs 2 shots

    Inputs:
            model -- FROMAGE model
            data -- open ended miniImage dataset
            caption -- how many previous captions to input
            image -- how many previous images to input
            recall -- represents k in recall

    Return: generated images and correponding story id
    """

    ## Flatten data to a 1d list
    # data = [i for story in open_ended_data for i in story]
    # print('data: ', list(data.keys()))
    print('length: ', len(open_ended_data))

    sample_examples = 500
    path = 'datasets/open_ended_mi/'
    prompts = []
    keys_for_prompt = []
    for i in range(15):
        if shot == 1:
            # keys_for_prompt = ['caption_1', 'image_1', 'caption_2', 'image_2', 'question', 'question_image']

            # for key_in_prompt in keys_for_prompt:
            #     if
            #     partial_prompt = open_ended_data[i][key_in_prompt]


            captions = [open_ended_data[i]['caption_1'], open_ended_data[i]['caption_2'], open_ended_data[i]['question']]
            # print('prompts: ', captions)
            img1 = utils.get_image_from_jpg(path=os.path.join(path, open_ended_data[i]['image_1']))
            img2 = utils.get_image_from_jpg(path=os.path.join(path, open_ended_data[i]['image_2']))
            img3 = utils.get_image_from_jpg(path=os.path.join(path, open_ended_data[i]['question_image']))
            imgs = [img1, img2, img3]
            # print('imgs: ', imgs)

            prompt = [captions[0], imgs[0], captions[1], imgs[1], captions[2], imgs[2]]
            prompts.append(prompt)

        elif shot == 2:
            ...


    # ## 1 shot (1 image of a blicket and 1 image of a dax) + (1 image with the question what it is)
    # ## => output should be dax/blicket
    # if shot == 1:
    #     prompts = [[f"{data[i][0]['original_text']}[RET]"] for i in range(4, len(data), 5)]  # size=(num_story*1)
    #     story_ids = [data[i][0]['story_id'] for i in range(4, len(data), 5)]  # size=(num_story*1)
    # ## 2 shots (2 images of a blicket and 2 images of a dax) + (1 image with the question what it is)
    # ## => output should be dax/blicket
    # elif shot == 2:
    #     ...
    # ## 5 captions and 4 images
    # else:
    #     prompts = []
    #     story_ids = []  # final size = (num_story*9)
    #     for i in range(0, len(data), 5):
    #         story_prompt = []
    #         for j in range(i, i + 5):
    #             if j != i + 4:
    #                 if 'url_o' in data[j][1].keys():
    #                     story_prompt.append(utils.get_image_from_url(data[j][1]['url_o']))
    #                 else:
    #                     story_prompt.append(utils.get_image_from_url(data[j][1]['url_m']))
    #                 story_prompt.append(data[j][0]['original_text'])
    #             else:
    #                     story_prompt.append(data[j][0]['original_text'])
    #         if len(story_prompt) == 9 and None not in story_prompt:
    #             prompts.append(story_prompt)
    #             story_ids.append(data[i][0]['story_id'])

    ## Inferecing
    model_outputs = []  # size=(num_story*recall)
    for i, prompt in enumerate(prompts):
        print('PROMPT: ', prompt)
        output = model.generate_for_images_and_texts(prompt, max_img_per_ret=recall, num_words=4)
        print('output: ', output)
        # if len(output[1]) == recall:
        #     model_outputs.append(output)
        # else:
        #     print(f"Not enough images")
        #     # story_ids[i] = None
    return model_outputs #, story_ids


## MAIN FUCNTION TO RUN EXPERIMENTS AND STORE OUTPUTS
def run_experiment(model, save_path: str, open_ended_data: list, shot: int = 1, recall: int = 1):
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
    model_outputs = generate_output(model=model, open_ended_data=open_ended_data, shot=1, recall=1)
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

    # new_tokens = ['dax', 'blicket']
    # model.model.tokenizer.add_tokens(new_tokens)
    # model.model.lm.resize_token_embeddings(len(model.model.tokenizer))

    print("-- Loading data:")
    # Load VIST dataset for experiment
    file_path = 'datasets/open_ended_mi/open_ended_mi_shots_1_ways_2_all_questions.json'
    with open(file_path, 'r') as f:
        open_ended_data = json.load(f)
    print(f"-- Finish loading | {len(open_ended_data)} stories")

    ## Define path to save results
    save_path = "\results"

    ## Define experiment configurations
    recall = [1, 5, 10]

    print(f"--- Experiment ongoing - 1 shot...")
    run_experiment(model=model, save_path=save_path, open_ended_data=open_ended_data, shot=1, recall=recall[0])
    print(f"--- Experiment finished")

    # else:
    #     print(f"--- Experiment ongoing - caption{config[0][0]} with 4 images...")
    #     run_experiment(model, save_path, VIST_data, config[0][0], config[0][1], config[1])
    #     print(f"--- Experiment finished")


__main__()
