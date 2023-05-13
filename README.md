# DL2_project
Reproduction of "Grounding Language Models to Images for Multimodal Generation"

# SETUP INSTRUCTIONS 

* pytorch==2.0.0
(to prevent the incompatibility issues with RTX titan running matrix multiplication on bf16)

# INSTRUCTIONS TO RUN THE EXAMPLE NOTEBOOK ON CLUSTER
Before doing the following steps, please make sure you have cloned the project to your cluster disk and installed all libraries in the requirments.txt (This is not added to the job file as it will run the installation everytime the job file runs, it might slow down a bit even though all requirements are already met after the first time).
After that, the following steps are necessary:
1. Modify line " " in final_notebook.ipynb to adapt to your own username and folder name.
3. Go into the directory where the job file is located. 
4. run sbatch run.job

# INSTRUCTIONS TO RUN THE FINAL NOTEBOOK





# INSTRUCTIONS TO RUN THE EXTENSION NOTEBOOK





# REPRODUCED RESULTS (TO BE CONTUNUED) : UPDATE THIS PART BEFORE SUBMITTING!
Results of experiments can be found in this link:
https://drive.google.com/drive/folders/1saV-XPLsoqL65xUEbOgrEhE7XbfL_WMJ?usp=share_link
