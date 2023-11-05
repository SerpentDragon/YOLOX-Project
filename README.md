# YOLOX Project

This program is a simple application for detecting objects on video using YOLOX library


## Installation

Step 1. Clone this repo.
```
git clone git@github.com:SerpentDragon/YOLOX-Project.git
```

Step 2. Install YOLOX from source to the repository that you downloaded in the previous step
```
cd YOLOX-Project
python3 -m venv your_venv_name
source your_venv_name/bin/activate
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
cd ..
```

**Warning!** You may need to install some additional modules to run the program. Errors can occur during the installation. For example, if you have problems with lzma module, try [https://github.com/ultralytics/yolov5/issues/1298](https://github.com/ultralytics/yolov5/issues/1298)

Step 3. You have to download pretrained model from [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

Step 4. Also, you have to download COCO dataset (annotation) from [https://cocodataset.org/#download](https://cocodataset.org/#download). Create COCO directory in YOLOX/datasets/ folder and unzip archive there

Step 5. Of course you should have video to process :))

Step 6. Now you can run the programm

**Warning!** You may need to use python3.10


## Options

Following options are available in the program:
1. --name (-n) - to specify name of the model you will use
2. --ckpt (-c) - to specify path to the model
3. --path (-p) - to specify path to your video file
4. --conf - to specify confidence threshold

If you do not specify confidence threshold value, it will be equal to 0!

**Run the program!**
```
python3 main.py -n model_name -c path/to/the/model -p path/to/video --conf value

```


## Demo

To run the programm you may use the following command:
```
python3.10 main.py -n yolox-x -c ./models/yolox_x.pth -p ./video/develop_streem.ts --conf 0.25
```

![Image alt](https://github.com/SerpentDragon/YOLOX-Project/blob/master/demo/demo.jpg)
