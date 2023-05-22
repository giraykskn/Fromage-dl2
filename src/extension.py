#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import json
from transformers import logging
import matplotlib.pyplot as plt
logging.set_verbosity_error()
import sys
sys.path.append('/home/lcur1748/Fromage-dl2')  ##change this to your folder where you put the whole project
from fromage import models
from fromage import utils
import logging
import time
import pickle
import argparse


# Create a logger
logger = logging.getLogger('my_logger')
# Set the logging level (optional)
logger.setLevel(logging.DEBUG)

class Experiment:
    def __init__(self, shot, way, repeat, use_sample=False):
        self.image_path = 'datasets/open_ended_mi/'
        logger.info(f"Running experiments with shot: {shot} and  way: {way}")
        path = f"datasets/open_ended_mi/open_ended_mi_shots_{shot}_ways_{way}_all_questions.json"
        with open(path, "r") as json_file:
            self.json = json.load(json_file)
        logger.info(f"Loaded the json file: {path}")
        self.use_sample = use_sample
        self.keys_for_prompt = Experiment.generate_keys(shot, way)
        self.repeats = repeat

    @staticmethod
    def generate_keys(shots, ways):
        return utils._generate_keys(shots,ways)

    def load_experiment(self):
        logger.info("Loading the experiments")
        self.prompts = []
        self.labels = []
        index = 15 if self.use_sample else len(self.json)
        for i in range(index):
            prompt = []
            # For each repeat, append it to the prompt.
            for repeat in range(self.repeats):
                for key_in_prompt in self.keys_for_prompt[:-1]:
                    # First append the image, then the caption
                    if 'caption' in key_in_prompt:
                        partial_prompt_text = self.json[i][key_in_prompt]
                        prompt.append(partial_prompt_text)
                    elif 'image' in key_in_prompt:
                        partial_prompt_image = utils.get_image_from_jpg(path=os.path.join(self.image_path, self.json[i][key_in_prompt]))
                        prompt.append(partial_prompt_image)
            
            # If it is the question, we need to ask a question, put the image, and then provide answering template
            partial_prompt_image = utils.get_image_from_jpg(path=os.path.join(self.image_path, self.json[i]["question_image"]))
            prompt.append(partial_prompt_image)
            prompt.append("Q: What is this?\nA: This is a")
            self.labels.append(self.json[i]['answer'])
            self.prompts.append(prompt)

def load_experiment(shots, ways, repeats):
    experiment = Experiment(shot = shots, way = ways, repeat = repeats, use_sample = False)
    experiment.load_experiment()
    return experiment

## FUNCTION OF MODEL RETRIEVING IMAGES
def generate_output(model, shots, ways, repeats, recall: int = 1):
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


    ## Inferecing
    experiment = load_experiment(shots = shots, ways = ways, repeats = repeats)
    model_outputs = []  # size=(num_story*recall)
    logger.info("Finished loading the experiment, inferencing with the fromage model")
    start_time = time.time()

    number_of_correct = 0
    for i, (prompt,label) in enumerate(zip(experiment.prompts, experiment.labels)):
        if (i % 50 == 0) and (i != 0):
            logger.warning(f"Accuracy after {i} examples is {number_of_correct / i}")
        model_outputs.append(prompt)
        output = model.generate_for_images_and_texts(prompt, max_img_per_ret=recall, num_words=2, temperature=0)
        model_outputs.append(output)
        logger.info("Output from model is : {}, target is : {}".format(output, label))
        number_of_correct += int(label in output[0])

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f'Finished the inference. Inference took {execution_time}')
    logger.warning(f"Out of {len(experiment.prompts)} model classified {number_of_correct} correctly. Accuracy is : {number_of_correct / len(experiment.prompts)}")
    return model_outputs


## MAIN FUCNTION TO RUN EXPERIMENTS AND STORE OUTPUTS
def run_experiment(model, save_path: str, shots: int = 1, ways: int = 2, repeats: int = 1, recall: int = 1):
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
    model_outputs = generate_output(model=model, shots=shots, ways=ways, repeats= repeats, recall=recall)
    ## Create path for the first time
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # TODO: make it save the prompt with the output (for now it only saves the npz files in the correct folder)
    ## Save results in npz file
    with open(f'{save_path}/extension_shots_{shots}_ways_{ways}_recall_{recall}_repeats_{repeats}.pkl', 'wb') as f:
        pickle.dump(model_outputs,f)


def __main__(number_of_ways, number_of_shots, number_of_repeats, file_name):
    # ### Load Model and Embedding Matrix
    # Load model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)
    # Define the logger to log experiments.
    file_handler = logging.FileHandler(file_name)
    logger.addHandler(file_handler)   
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    ## Define path to save results
    save_path = "Results/Extension/"

    ## Define experiment configurations
    recall = [1, 5, 10]

    # TODO: make commands to run all combinations of experiments
    logger.info(f"--- Experiment ongoing - 1 shot...")
    run_experiment(model=model, save_path=save_path, shots=number_of_shots, ways=number_of_ways,repeats = number_of_repeats, recall=recall[0])
    logger.info(f"--- Experiment finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process number of ways and number of shots.")
    parser.add_argument("-w", "--ways", type=int, help="Number of ways")
    parser.add_argument("-s", "--shots", type=int, help="Number of shots")
    parser.add_argument("-r", "--repeats", type=int, help="Number of repeats")
    parser.add_argument("-f", "--file", type=str, help="File name to save the log")
    args = parser.parse_args()
    __main__(args.ways, args.shots, args.repeats, args.file)