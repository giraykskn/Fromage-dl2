#                                DL2_project
Reproduction of "Grounding Language Models to Images for Multimodal Generation"

## SETUP INSTRUCTIONS 
* pytorch==2.0.0 installed
(to prevent the incompatibility issues with RTX titan running matrix multiplication on bf16)

## MODIFICATION OF ORIGINAL FILES
***models.py***: 
1. added bfloat16 parameter when loading model:
```
self.lm = OPTForCausalLM.from_pretrained(opt_version, torch_dtype=torch.bfloat16)
self.visual_model = CLIPVisionModel.from_pretrained(visual_encoder, torch_dtype=torch.bfloat16)
```
2. added torch.nn.DataParallel to enable multiple GPU (4 GPU) use case (currently commented out since we only use one GPU for now):
```
self.lm = torch.nn.DataParallel(self.lm, device_ids=[0, 1, 2, 3])
self.visual_model = torch.nn.DataParallel(self.visual_model, device_ids=[0, 1, 2, 3])
```
***util.py***:
added `timeout=5` and `try & except` in `get_image_from_url` function for the same reason as some urls might not be responding therefore, to prevent model from running forever.
if link is invalid, function returns None:
```
## Modified function
def get_image_from_url(url: str):
    try:
      response = requests.get(url, timeout=5)
      img = Image.open(BytesIO(response.content))
      img = img.resize((224, 224))
      img = img.convert('RGB')
      return img
    except:
      return None
```


## INSTRUCTIONS TO REPRODUCTION EXPERIMENTS
The experiment we choose to reproduce is Visual Story Telling (see section 4.1 in the paper). VIST dataset used can be found in https://visionandlanguage.net/VIST/dataset.html (Stories of
Images-in-Sequence (SIS)). We preprossed this dataset and converted into a json file named `VIST_data_for_experiments.json` which can be found in `src` folder. The main file to run the experiments 
is named `reproduce.py` and `reproduce.job` is the job file used to run the py file in cluster, they can be found in src folder as well.It includes in total 5050 story sequences and each story 
sequence has 5 images and 5 corresponding story description which form a short story:
```
image_1 - story_1
image_2 - story_2
image_3 - story_3
image_4 - story_4
last_image - story_5
```
The goal of this experiment is to predict the last image based on a few input combinations of previous information (previous images and stories). The following 3 input settings are experimented:
```
1. story_4
2. story_1 + story_2 + story_3 + story_4 + story_5
3. story_1 + image_1 + story_2 + image_2 + story_3 + image_3 + story_4 + image_4 + story_5
```
For each input setting, three different recall levels are also experimented:
```
recall@1 ; recall@5 ; recall@10
```

## INSTRUCTIONS TO RUN THE FINAL NOTEBOOK





## INSTRUCTIONS TO RUN THE EXTENSION NOTEBOOK
First, the dataset that was used in the extension from the paper " Multimodal Few-Shot Learning with Frozen Language Model " needs to be downloaded. The "open_ended_mi" dataset can be found in the following page: https://fh295.github.io/frozen.html

Next steps:
1. Place the ***"open_ended_mi.tar.gz"*** dataset file inside ***datasets*** folder , found in the ***src*** folder. 
2. Decompress the tar file inside the ***datasets*** folder. 
3. Adjust the path to the own root directory in ***extension.py***, line 13.
4. Run the job file for the extension (***extension.job***).



## REPRODUCED RESULTS (TO BE CONTUNUED) : UPDATE THIS PART BEFORE SUBMITTING!
Results (recall) of reproduced experiments are as follows:
```
Caption 1 recall 1 Image False: 0.15
Caption 1 recall 5 Image False: 0.22
Caption 1 recall 10 Image False: 0.29
Caption 5 recall 1 Image False: 0.22
Caption 5 recall 5 Image False: 0.31
Caption 5 recall 10 Image False: 0.41
Caption 5 recall 1 Image True: 0.20
Caption 5 recall 5 Image True: 0.31
Caption 5 recall 10 Image True: 0.41
```
*E.g. Caption 1 recall 1 Image False: inputs with only 1 caption without any images evaluated at Recall@1*
Recalls for inputs with images are not as expected. They should have been higher than all other scores. 
