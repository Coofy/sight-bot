# sight-bot

## Setup Windows
```
git clone --depth 1 https://github.com/tensorflow/models
cd models/research/
for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.
pip install .
```
