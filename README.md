# DL2_project
Reproduction of "Grounding Language Models to Images for Multimodal Generation"

## SETUP INSTRUCTIONS 
* Model is loaded in bfloat16 because of memory insufficiency on GPU;
* Dataparallelization is added for multiple GPU computation (currently commented out since we only use one GPU for now);
* pytorch==2.0.0 installed
(to prevent the incompatibility issues with RTX titan running matrix multiplication on bf16)

## MODIFICATION OF ORIGINAL FILES
util.py: added timeout and try & except in 'get_image_from_url' function for the same reason as some urls might not be responding therefore, to prevent model from running forever.

# INSTRUCTIONS TO RUN REPRODUCTION EXPERIMENTS
The experiment we choose to reproduce is Visual Story Telling (see section 4.1 in the paper). VIST dataset used can be found in https://visionandlanguage.net/VIST/dataset.html (Stories of
Images-in-Sequence (SIS)). It includes in total 5050 stories and each story has 5 images and 5 corresponding captions which form a short story. We preprossed this dataset and converted
into a json file named 'VIST_data_for_experiments.json' which can be found in src folder. The main file to run the experiments is named 'reproduce.py' and 'reproduce.job' is the job file 
used to run the py file in cluster, they can be found in src folder as well.
3 experimental settings: 
1. retrieve last image in a story using 1 caption; 
2. retrieve last image using 5 captions;
3. retrieve last image using 5 captions and 4 previous images

# INSTRUCTIONS TO RUN THE FINAL NOTEBOOK





# INSTRUCTIONS TO RUN THE EXTENSION NOTEBOOK





# REPRODUCED RESULTS (TO BE CONTUNUED) : UPDATE THIS PART BEFORE SUBMITTING!
Results of reproduced experiments can be found in this link:
https://drive.google.com/drive/folders/1saV-XPLsoqL65xUEbOgrEhE7XbfL_WMJ?usp=share_link
