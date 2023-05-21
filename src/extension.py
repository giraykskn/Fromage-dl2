#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import json
from transformers import logging
import matplotlib.pyplot as plt

logging.set_verbosity_error()

import sys

sys.path.append('/home/lcur1734/Fromage-dl2')  ##change this to your folder where you put the whole project

from fromage import models
from fromage import utils


## FUNCTION OF MODEL RETRIEVING IMAGES
def generate_output(model, shots: int = 1, ways: int = 2, recall: int = 1):
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

    sample_examples = 500
    image_path = 'datasets/open_ended_mi/'
    json_path = ''

    prompts = []
    keys_for_prompt = []

    # TODO: include the other json files as json_path and keys_for_prompt => add other if-statements

    # ATTENTION: the keys_for_prompt should be in the format 'caption, image, caption, image...' with
    # 'question' and 'question_image' at the end

    if shots == 1 and ways == 2:
        print('running 1 shot, 2 ways')
        json_path = 'datasets/open_ended_mi/open_ended_mi_shots_1_ways_2_all_questions.json'
        keys_for_prompt = ['caption_1', 'image_1',
                           'caption_2', 'image_2',
                           'question_image']
    elif shots == 1 and ways == 5:
        print('running 1 shot, 5 ways')
        json_path = 'datasets/open_ended_mi/open_ended_mi_shots_1_ways_5_all_questions.json'
        keys_for_prompt = ['caption_1', 'image_1',
                           'caption_2', 'image_2',
                           'question_image']
    elif shots == 3 and ways == 2:
        print('running 3 shot, 2 ways')
        json_path = 'datasets/open_ended_mi/open_ended_mi_shots_3_ways_2_all_questions.json'
        keys_for_prompt = ['caption_1', 'image_1',
                           'caption_2', 'image_2',
                           'caption_3', 'image_3',
                           'caption_4', 'image_4',
                           'caption_5', 'image_5',
                           'caption_6', 'image_6',
                           'question_image']
    elif shots == 3 and ways == 5:
        print('running 3 shot, 5 ways')
        json_path = 'datasets/open_ended_mi/open_ended_mi_shots_3_ways_5_all_questions.json'
        keys_for_prompt = ['caption_1', 'image_1',
                           'caption_2', 'image_2',
                           'caption_3', 'image_3',
                           'caption_4', 'image_4',
                           'caption_5', 'image_5',
                           'caption_6', 'image_6',
                           'question_image']
    elif shots == 5 and ways == 2:
        print('running 5 shots, 2 ways')
        json_path = 'datasets/open_ended_mi/open_ended_mi_shots_5_ways_2_all_questions.json'
        keys_for_prompt = ['caption_1', 'image_1',
                           'caption_2', 'image_2',
                           'caption_3', 'image_3',
                           'caption_4', 'image_4',
                           'caption_5', 'image_5',
                           'caption_6', 'image_6',
                           'caption_7', 'image_7',
                           'caption_8', 'image_8',
                           'caption_9', 'image_9',
                           'caption_10', 'image_10',
                           'question_image']
    elif shots == 5 and ways == 5:
        print('running 5 shots, 5 ways')
        json_path = 'datasets/open_ended_mi/open_ended_mi_shots_5_ways_5_all_questions.json'
        keys_for_prompt = ['caption_1', 'image_1',
                           'caption_2', 'image_2',
                           'caption_3', 'image_3',
                           'caption_4', 'image_4',
                           'caption_5', 'image_5',
                           'caption_6', 'image_6',
                           'caption_7', 'image_7',
                           'caption_8', 'image_8',
                           'caption_9', 'image_9',
                           'caption_10', 'image_10',
                           'question_image']
    else:
        raise Exception('Incorrect number of shots or ways')

    with open(json_path, 'r') as f:
        open_ended_data = json.load(f)

    # TODO: make this for loop take random samples (the amount of sample_examples) since now it takes just the first 500

    # TODO: set seed so it takes the same random samples every time
    for i in range(15):
        prompt = []
        for key_in_prompt in keys_for_prompt:
            if ('caption' in key_in_prompt) or (key_in_prompt == 'question'):
                partial_prompt_text = open_ended_data[i][key_in_prompt]
                prompt.append(partial_prompt_text)

            else:
                partial_prompt_image = utils.get_image_from_jpg(path=os.path.join(image_path, open_ended_data[i][key_in_prompt]))
                prompt.append(partial_prompt_image)

        prompts.append(prompt)

    ## Inferecing
    model_outputs = []  # size=(num_story*recall)
    for i, prompt in enumerate(prompts):
        model_outputs.append(prompt)
        output = model.generate_for_images_and_texts(prompt, max_img_per_ret=recall, num_words=256, temperature=1.5)
        model_outputs.append(output)
        print('output: ', output)

    return model_outputs


## MAIN FUCNTION TO RUN EXPERIMENTS AND STORE OUTPUTS
def run_experiment(model, save_path: str, shots: int = 1, ways: int = 2,  recall: int = 1):
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
    model_outputs = generate_output(model=model, shots=shots, ways=ways, recall=recall)
    ## Create path for the first time
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # TODO: make it save the prompt with the output (for now it only saves the npz files in the correct folder)
    ## Save results in npz file
    with open(f'{save_path}/extension_shots{shot}_ways{ways}_recall{recall}.npz', 'wb') as f:
        np.savez(f, images=model_outputs)


def __main__():
    # ### Load Model and Embedding Matrix
    # Load model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)

    ## Define path to save results
    save_path = "Results/Extension/"

    ## Define experiment configurations
    recall = [1, 5, 10]

    # TODO: make commands to run all combinations of experiments
    print(f"--- Experiment ongoing - 1 shot...")
    run_experiment(model=model, save_path=save_path, shots=5, ways=2, recall=recall[0])
    print(f"--- Experiment finished")



__main__()
