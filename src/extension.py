#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import json
from transformers import logging
import itertools
import matplotlib.pyplot as plt

logging.set_verbosity_error()

import sys

sys.path.append('/home/lcur1734/Fromage-dl2')  ##change this to your folder where you put the whole project

from fromage import models
from fromage import utils



## FUNCTION OF MODEL RETRIEVING IMAGES
def generate_output(model, shots: int = 1, ways: int = 2):
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

    image_path = 'datasets/open_ended_mi/'
    json_path = ''

    prompts = []
    keys_for_prompt = []
    labels = []
    inputs = []


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
    for i in range(10):
        prompt = []
        for key_in_prompt in keys_for_prompt:
            if ('caption' in key_in_prompt) or (key_in_prompt == 'question'):
                partial_prompt_text = open_ended_data[i][key_in_prompt]
                prompt.append(partial_prompt_text)

            else:
                partial_prompt_image = utils.get_image_from_jpg(path=os.path.join(image_path, open_ended_data[i][key_in_prompt]))
                prompt.append(partial_prompt_image)

        prompts.append(prompt)
        labels.append(open_ended_data[i]["answer"])
        inputs.append(open_ended_data[i]["question_id"])
    print(labels)

    ## Inferecing
    generated = []
    for i, prompt in enumerate(prompts):
        print("-------------" + str(i + 1) + "---------------")
        output = model.generate_for_images_and_texts(prompt, max_num_rets=0,  num_words=5, temperature=0)
        generated.append(output)
        print('output: ', output)
    correct = 0
    for i, label in enumerate(labels):
        if label in generated[i][0]:
            correct += 1
    print(f"Accuracy for {shots} shots and {ways} ways: ", correct / len(generated))
    return inputs, generated, labels


## MAIN FUCNTION TO RUN EXPERIMENTS AND STORE OUTPUTS
def run_experiment(model, save_path: str, shots: int = 1, ways: int = 2):
    """
    Inputs:
            model -- FROMAGE model
            save_path -- path to save results
            data -- story sequences from open-mi
            shots -- how many shots of images
            ways -- how many different categories of images used

    Return: generated images and correponding story id
    """
    inputs, generated, labels = generate_output(model=model, shots=shots, ways=ways)
    ## Create path for the first time
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # TODO: make it save the prompt with the output (for now it only saves the npz files in the correct folder)
    ## Save results in npz file
    with open(f'{save_path}/extension_shots{shots}_ways{ways}.npz', 'wb') as f:
        np.savez(f, inputs=np.array(inputs), generated=np.array(generated), labels=np.array(labels))


def __main__():
    # ### Load Model and Embedding Matrix
    # Load model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)

    ## Define path to save results
    save_path = "Results/Extension/"

    ## Define experiment configurations
    shots = [1, 3, 5]
    ways = [2, 5]
    config_combinations = list(itertools.product(shots, ways))

    # Test experiment
    """print(f"--- Experiment ongoing - 1 shot...")
    run_experiment(model=model, save_path=save_path, shots=5, ways=2)
    print(f"--- Experiment finished")"""
    
    # Run all experiments
    for config in config_combinations:
        print(f"--- Experiment ongoing - {config[0]} shots with {config[1]} ways...")
        run_experiment(model=model, save_path="extension_results", shots=config[0], ways=config[1])
        print(f"--- Experiment finished")



__main__()
