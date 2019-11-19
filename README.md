# Sight Bot

## Setup
```
Miniconda:

conda create -n perception-tf1.15 pip python=3.7 pyqt=5
```

### Tensorflow
```
pip install tensorflow==1.15
```

### object_detection
```
git clone --depth 1 https://github.com/tensorflow/models

cd models/research/
protoc object_detection/protos/*.proto --python_out=.
pip install .
```

### OpenCV
```
pip install opencv-python
```

### LabelImg
```
cd addons/labelImg

sudo apt-get install pyqt5-dev-tools
sudo pip install -r requirements/requirements-linux-python3.txt
make qt5py3
```

## Training
### Single/Poor GPU
```
TF_ENABLE_GPU_GARBAGE_COLLECTION=false TF_FORCE_GPU_ALLOW_GROWTH=true python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
```

## Running
```
python main.py

### Force CPU
CUDA_VISIBLE_DEVICES=-1 python main.py
```
