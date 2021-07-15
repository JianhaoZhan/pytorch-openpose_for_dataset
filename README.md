# openpose_for_dataset

extract pic with skeleton by pytorch-openpose(https://github.com/Hzzone/pytorch-openpose)

the environment installation and model downloading refers https://github.com/Hzzone/pytorch-openpose

To extract HMDB51 with pytorch-openpose:

# PATH_ORG : the path of your dataset(RGB)   PATH_TARGET : the path of your extracted dataset to save

# HMDB51

python rebuild_HMDB51.py   PATH_ORG   PATH_TARGET

# UCF101

python rebuild_UCF101.py   PATH_ORG   PATH_TARGET

# NTU-RGB-D

python rebuild_ntu.py      PATH_ORG   PATH_TARGET
