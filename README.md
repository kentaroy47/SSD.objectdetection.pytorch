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
https://github.com/kentaroy47/ObjectDetection.Pytorch/releases/download/ssd200/ssd300_200.pth

## Train YOLO Models
run yolo.ipynb (TBD)

## Train Faster RCNN Models
run frcnn.ipynb (TBD)
