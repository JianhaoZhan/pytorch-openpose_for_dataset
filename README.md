# openpose_for_dataset

extract pic with skeleton by [pytorch-openpose](https://github.com/Hzzone/pytorch-openpose)

the environment installation and model downloading refers [environment](https://github.com/Hzzone/pytorch-openpose)

# To extract skeleton with pytorch-openpose:

flags: PATH_ORG : the path of your dataset(RGB); PATH_TARGET : the path of your extracted dataset to save.

## HMDB51
```python
  python rebuild_HMDB51.py   PATH_ORG   PATH_TARGET
```
## UCF101
```python
  python rebuild_UCF101.py   PATH_ORG   PATH_TARGET
```
## NTU-RGB-D
```python
  python rebuild_ntu.py      PATH_ORG   PATH_TARGET
```
  ![example](https://github.com/JianhaoZhan/pytorch-openpose_for_dataset/blob/main/example.jpg)

  you can alse extract using the .skeleton file provided by [NTU-RGB-D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) to get RGB files with skeleton, and the figure as follows :
  
  ![example](https://github.com/JianhaoZhan/pytorch-openpose_for_dataset/blob/main/others.jpg)

  if you want to get that, you can follow our another repository :[ntu_extract_org](https://github.com/JianhaoZhan/ntu_extract_org)
