# arcface-embedder
Simple script for face detection, alignment and getting embeddings using pretrain models from [insightface project](https://github.com/deepinsight/insightface.git), converted to PyTorch format using [pytorch-insightface project](https://github.com/nizhib/pytorch-insightface.git) and using detector from [MTCNN](https://github.com/faciallab/FaceDetector.git) project. Face alignment implemented using pytorch Tensor computing, based on original [insightface numpy realization](https://github.com/deepinsight/insightface/blob/master/recognition/common/face_align.py)


# Install/build dependencies (FaceDetector and pytorch-insightface)
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
$ cd FaceDetector/
$ pip install opencv-python numpy easydict Cython progressbar2 torch tensorboardX
$ python setup.py build_ext --inplace
$ python setup.py install
```

3. Install insightface module:
```sh
$ pip install git+https://github.com/nizhib/pytorch-insightface
```

4. If you want to get weights for embedder locally, you need to convert weights from [model zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo), 3.1, 3.2 and 3.3, follow instructions from `pytorch-insightface/README.md`: download the original insightface zoo weights and place *.params and *.json files to pytorch-insightface/resource/{model}.
Then run python `pytorch-insightface/scripts/convert.py` to convert and test pytorch weights.

# Usage
You can just run script without any parameters, all available input parameters will be setted to default values, see next section for details:
```sh
$ python3 run.py
```

For running with specific image path:
```sh
$ python3 run.py --image-path [image name]
```

# Options
There are some input parameters available:
* --image-path: path to image to be processed. Default: ./images/office5.jpg
* --is-local-weights: whether to use local weights or from remote server. Default: 0
* --weights-base-path: root path to insightface weights, converted to PyTorch format.
Actual only if --is-local-weights == 1. Default: pytorch-insightface/resource
* --show-face: whether to show cropped and aligned face or not. Default: 0
* --align-torch: whether to use torch or numpy realization for face alignment. Default: 1
* --arch: architecture of embedder: iresnet34|iresnet50|iresnet100. Default: iresnet100

# Some examples of aligning and distance between features
For comparing two vectors of features L1 norm was used: 
```python
numpy.linalg.norm(v1-v2, 1)
```
where `v1` and `v2` are feature vectors of size 512

detected original face | aligned image 1 | aligned image 2 | L1-norm
-----------------------|-----------------|-----------------|----------
![Person 1 face orig](/images/out/per1_det.jpg) | torch align: <br/> ![Person 1 align torch](/images/out/per1_torch.jpg) | numpy align: <br/> ![Person 1 align numpy](/images/out/per1_numpy.jpg) | 2.4723
![Person 2 face orig](/images/out/per2_det.jpg) | torch align: <br/> ![Person 2 align torch](/images/out/per2_torch.jpg) | numpy align: <br/> ![Person 2 align numpy](/images/out/per2_numpy.jpg) | 2.3088
![Person 3 face orig](/images/out/per3_det.jpg) | torch align: <br/> ![person 3 align torch](/images/out/per3_torch.jpg) | numpy align: <br/> ![Person 3 align numpy](/images/out/per3_numpy.jpg) | 1.5965
--- | torch align: <br/> ![Person 1 align torch](/images/out/per1_torch.jpg) | torch align: <br/> ![Person 2 align torch](/images/out/per2_torch.jpg) | 29.0398
--- | torch align: <br/> ![Person 1 align torch](/images/out/per1_torch.jpg) | torch align: <br/> ![Person 3 align torch](/images/out/per3_torch.jpg) | 23.1007
--- | torch align: <br/> ![Person 2 align torch](/images/out/per2_torch.jpg) | torch align: <br/> ![Person 3 align torch](/images/out/per3_torch.jpg) | 27.0426
 
 

