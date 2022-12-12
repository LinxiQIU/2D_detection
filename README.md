# 2d_detection

This part is to illustrate how to convert the generated dataset in COCO format into a process that can be directly trained with YOLOv5. For how to generate the dataset, please use this [guide](https://github.com/LinxiQIU/Motor_Datasets_Generation).

# Before You Start 

Clone repo and install [YOLOv5](https://github.com/ultralytics/yolov5), creating a custom model to detect your objects is an iterative process of collecting and organizing images, labeling your objects of interest, training a model.


According to the instructions for training custom data in [YOLOv5](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data), we can prepare the dataset manually.

## 1.1 Create dataset.yaml 

The dataset config file that defines 1) the dataset root directory `path` and relative paths to `train`/`val`/`test` image directories (or *.txt files with image paths) and 2) a class `names` dictionary:
```python
path: ../dataset/motor1000_coco  # dataset root dir
train: images
val: images
test:  #

# Classes
nc: 3   # number of classes
names: ['side_screw', 'cover_screw', 'motor']
```

## 1.2 Create Labels 
**YOLO format** ist one `*.txt` file per image. The `*.txt` file specifications are:

* One row per object.
* Each row is `class x_center y_center width height` format.
* Box coordinates must be in **normalized xywh** format (from 0-1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.
* Class numbers are zero-indexed (start from 0).

<img src="https://github.com/LinxiQIU/2d_detection/blob/main/images/label.png" width="640" height="480">

The label file corrsponding to the image contains one motor, two side screws and three cover screws.

## 1.3 Organize Directories 

Organize your train and val images and labels according to the example below. **YOLOv5 locates labels automatically for each image** by replacing the last instance of `/images/` in each image path `/labels/`. For example:

```python
../datasets/images/im0.jpg  # image
../datasets/labels/im0.txt  # label
```
You can use the script `raw2coco.py` to save each image and the corresponding `json` file directly in **YOLO format**.

## 2. Select a Model
Select a pretrained model to start training from. There are five sizes of model.

## 3. Train

Train a YOLOv5s model on motor dataset, batch-size, image size and either pretrained `--weights yolov5s.pt` (recommended), or randomly initialized `--weights '' --cfg yolov5s.yaml` (not recommended). Pretrained weights are auto-downloaded from the latest YOLOv5 release.

```python
# Train YOLOv5s on Motor_1000 for 100 epochs
$ CUDA_VISIBLE_DEVICES=1 python train.py --img 1280 --batch 16 --epochs 100 --data motor1000_coco.yaml --weights yolov5s.pt
```

## 4 Test Result

Here is the result of the test with trained model:
<img src="https://github.com/LinxiQIU/2d_detection/blob/main/images/yolo_result.png" width="960" height="540">