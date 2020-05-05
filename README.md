# SOFT Tracker with YoLOv3 detector

Tracker based on simple online and realtime tracking algorithm [paper](https://arxiv.org/abs/1602.00763) and it implemenation open source. 
Detector to tracking is YOLOv3.
In this repo is only human tracking

Tested on [Campus sequences](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)

## Requirements
1. Python >= 3.5 
2. OpenCV >= 3.4.0 
3. PyTorch >=0.4
4. filterpy 1.4.5 ```pip install filterpy```


## Running the tracker

### SOFT tracker with YOLOv3 detector


Command to human tracking 
```
python tracker.py --path-to-video video.avi
```

