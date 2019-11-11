# sight-bot

Based on this repository: [object_detection_tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)

## Setup Windows

* CUDA: v10.0
* cuDNN: v7.6.5.32
* Tensorflow v2.0.0

```
git clone --depth 1 https://github.com/tensorflow/models
cd models/research/
for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.
pip install .
```
