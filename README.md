# ObjectDetection.Pytorch
![teaser](https://github.com/kentaroy47/ObjectDetection.Pytorch/blob/master/imgs/1.png)

This repo is an object detection library for pytorch (ssd, yolo, faster-rcnn).

## To start off
requirements: cv2, pandas. plz install.

clone the repo.
```
git clone https://github.com/kentaroy47/ObjectDetection.Pytorch.git
```

Download PASCALVOC2007 dataset and extract.
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
```

Download reduced FC vgg weights and place in weights folder.
```
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

## Train SSD Models
run train_ssd.ipynb

to run inference, try inference.ipynb.

the trained SSD model is here (still underfitting..)
https://github.com/kentaroy47/ObjectDetection.Pytorch/releases/download/ssdvgg200/ssd300_200.pth

## Train YOLO Models
run yolo.ipynb (TBD)

## Train Faster RCNN Models
run frcnn.ipynb (TBD)

## Test models
run `eval.ipynb`

# Test results for ssd

Pascal VOC 2007 test set.

### SSD-300
Model:
https://github.com/kentaroy47/ObjectDetection.Pytorch/releases/download/ssdvgg200/ssd300_200.pth

Mean AP = 0.7959

~~~~~~~~
Results:
0.842
0.850
0.784
0.736
0.518
0.891
0.888
0.902
0.634
0.832
0.793
0.873
0.899
0.862
0.815
0.521
0.798
0.815
0.885
0.780
0.796
~~~~~~~~

### RetinaNet-300 Resnet18
Mean AP = 0.7279
~~~~~~~~
Results:
0.759
0.814
0.725
0.661
0.373
0.807
0.836
0.847
0.508
0.759
0.741
0.816
0.848
0.813
0.743
0.420
0.695
0.803
0.859
0.731
0.728
~~~~~~~~
