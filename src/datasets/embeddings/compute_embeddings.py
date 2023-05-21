repo_path = '/home/lcur1748/Fromage-dl2'
import sys
import os
sys.path.append(os.path.join(repo_path, 'src'))
from fromage import models
from fromage import utils
import glob
import torch
from PIL import Image
import json
import pickle

image_path = os.path.join(repo_path, 'src/datasets/open_ended_mi/')
model_dir  = os.path.join(repo_path, 'src/fromage_model')
path_to_json = os.path.join(repo_path,'src/datasets/open_ended_mi_shots_1_ways_2_all_questions.json')

print(image_path)
print(model_dir)
print(path_to_json)
def _compute_embeddings_from_json(args, path_to_json = path_to_json, nr_of_images = 2):
    # Load the model
    model = models.load_fromage(model_dir)
    images = [f'image_{i}' for i in range(1,nr_of_images)]
    print("images: {}", images)
    embeddings = {'embeddings': [], 'paths': []}
    # Open the json file
    with open(path_to_json,'r') as json_file_:
        json_file = json.load(json_file_)
        print("Loaded json file")
        # Loop through experiments and get the images
        for experiment in json_file:
            for image in images:
                # Get the image path
                print("Embedding the image : {}".format(experiment[image]))
                path = os.path.join(image_path,experiment[image])
                # Get the embeddings from the model.
                p = utils.get_image_from_jpg(path=path)
                pixel_values = utils.get_pixel_values_for_model(model.model.feature_extractor, p)
                pixel_values = pixel_values.to(device=model.model.logit_scale.device, dtype=model.model.logit_scale.dtype)
                pixel_values = pixel_values[None, ...]
                visual_embs = model.model.get_visual_embs(pixel_values, mode='retrieval')
                embeddings['embeddings'].append(visual_embs)
                # This means that we will run the model from the /src directory!!
                embeddings['paths'].append('datasets/embeddings/open_ended_mi/{}'.format(experiment[image]))
                print("Finished embedding : {}, saved the path as : {}".format(experiment[image], 'datasets/embeddings/open_ended_mi/{}'.format(experiment[image])))
    _, tail = os.path.split(path_to_json)
    name, _ = os.path.splitext(tail)
    save_path = os.path.join(repo_path, 'src/datasets/embeddings/{}.pkl'.format(name))
    print("Saving the pickle to : {}".format(save_path))
    with open(save_path, 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    _compute_embeddings_from_json(None)