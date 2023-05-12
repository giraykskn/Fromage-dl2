# DL2_project
Reproduction of "Grounding Language Models to Images for Multimodal Generation"

# HOW TO RUN INFERENCE ON CLUSTER
Before doing the following steps, please make sure you have cloned the project to your cluster disk and installed all libraries in the requirments.txt (I didn't add this to the job file because it will run the installation everytime you run the job file, it might slows down a bit even though after the first time, all requirements are already met).
After that, please do the following steps:
1. Overwrite model.py using the one in the respository. Some changes:
   1) Load model using bf16 to reduce memory otherwise GPU runs out of memory (when using 1 GPU)
   2) Added DataParallel for multiple GPU case but for now it is commented out because it seems we can only use 1 GPU 
      when directly running jobs on cluster
2. Put FROMAGe_example_notebook_python.py to the main folder. It is python version of the inference jupyter notebook.
3. Modify line 25 in FROMAGe_example_notebook_python.py to adapt to your own username and folder name.
4. Upgrade torch before running the job because otherwise RTX titan is not compatible to run matrix multiplication on    bf16 (at least in my environment).
5. Forward to your main folder before running the job 
6. run sbatch run.job
   * There are some other parameters that can be added to the job file, please take a look      https://servicedesk.surf.nl/wiki/display/WIKI/SLURM+batch+system#SLURMbatchsystem-Definingtherequirementsofajob if        you are interested.

* For now, the display function maintains the same but we can change it to have all the images saved in another folder
* There might be some issues with this set-up since our running environments are different.
* The whole process took around 4 mins so not too bad

# MODIFICATION TO THE ORIGINAL FILES
1. model.py: 
   1) added try & except to line 578-588 because some of the urls are not invalid therefore, if not modified, inference either runs forever or suffer from connection failure.
   2) changed max_img_per_ret + 3 to max_img_per_ret + 6 because in the case of 3, model sometimes returs number of images less than the value of max_img_per_ret which causes error when saving images in npz file.
3. util.py: added timeout and try & except in get_image_from_url function for the same reason as 1. Some urls might not be responding therefore, to prevent 
   model from running forever.

# REPRODUCED RESULTS (TO BE CONTUNUED)
Results of experiments can be found in this link:
https://drive.google.com/drive/folders/1saV-XPLsoqL65xUEbOgrEhE7XbfL_WMJ?usp=share_link
