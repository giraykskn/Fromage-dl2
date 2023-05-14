# DL2_project
Reproduction of "Grounding Language Models to Images for Multimodal Generation"

# SETUP INSTRUCTIONS 
* Model is loaded in bfloat16 because of memory insufficiency on GPU;
* Dataparallelization is added for multiple GPU computation (currently commented out since we only use one GPU for now);
* pytorch==2.0.0 installed
(to prevent the incompatibility issues with RTX titan running matrix multiplication on bf16)

# MODIFICATION OF ORIGINAL FILES
* util.py: added timeout and try & except in 'get_image_from_url' function for the same reason as some urls might not be responding therefore, to prevent model from running forever.

# INSTRUCTIONS TO RUN REPRODUCTION EXPERIMENTS
* 

# INSTRUCTIONS TO RUN THE FINAL NOTEBOOK





# INSTRUCTIONS TO RUN THE EXTENSION NOTEBOOK





# REPRODUCED RESULTS (TO BE CONTUNUED) : UPDATE THIS PART BEFORE SUBMITTING!
Results of experiments can be found in this link:
https://drive.google.com/drive/folders/1saV-XPLsoqL65xUEbOgrEhE7XbfL_WMJ?usp=share_link
