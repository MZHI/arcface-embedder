# arcface-embedder
Simple script for face detection and embedding using pretrain models from [insightface project](https://github.com/deepinsight/insightface.git), converted to PyTorch format using [this project](https://github.com/nizhib/pytorch-insightface.git) and using detector from [MTCNN](https://github.com/faciallab/FaceDetector.git) project.


# Install/build dependencies
1. Load submodules:
```sh
$ git submodule init
```
or just use one-step method during clonnig project:
```sh
$ git clone --recursive https://github.com/MZHI/arcface-embedder.git
```

2. Build FaceDetector submodule using instructions from `FaceDetector/README.md`:
```sh
$ pip install opencv-python numpy easydict Cython progressbar2 torch tensorboardX
$ python setup.py build_ext --inplace
$ python setup.py install
```

3. Install insightface module:
```sh
$ pip install git+https://github.com/nizhib/pytorch-insightface
```

4. If you want to get weights for embedder locally, you need to convert weights from [model zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo), 3.1, 3.2 and 3.3, follow instructions from `pytorch-insightface/README.md`: download the original insightface zoo weights and place *.params and *.json files to pytorch-insightface/resource/{model}.
Then run python pytorch-insightface/scripts/convert.py to convert and test pytorch weights.

# Usage
You can run script without any parameters:
```sh
$ python3 run.py
```

Also, there are some input parameters available:
* --image-path: path to image to be processed. Default: ./images/office5.jpg
* --is-local-weights: whether to use local weights or from remote server. Default: 0
* --weights-base-path: root path to insightface weights, converted to PyTorch format.
Actual only if --is-local-weights == 1. Default: pytorch-insightface/resource
* --show-face: whether to show cropped face or not. Default: 0
