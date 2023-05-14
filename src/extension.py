#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import json
from transformers import logging

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
        json_path = 'datasets/open_ended_mi/open_ended_mi_shots_1_ways_2_all_questions.json'
        keys_for_prompt = ['caption_1', 'image_1', 'caption_2', 'image_2', 'question', 'question_image']
    elif shots == 2:
        ...

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
        output = model.generate_for_images_and_texts(prompt, max_img_per_ret=recall, num_words=4)
        print('output: ', output)

    return model_outputs


## MAIN FUCNTION TO RUN EXPERIMENTS AND STORE OUTPUTS
def run_experiment(model, save_path: str, shot: int = 1, ways: int = 2,  recall: int = 1):
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
    model_outputs = generate_output(model=model, shots=1, ways=2, recall=1)
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
    run_experiment(model=model, save_path=save_path, shot=1, ways=2, recall=recall[0])
    print(f"--- Experiment finished")



__main__()
