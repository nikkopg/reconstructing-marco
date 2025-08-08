# Reconstructing MARCO Model
This repository contains a simple reconstruction of Marco Model from "[Automating the Evaluation of Crystallization Experiments](https://arxiv.org/pdf/1803.10342)" paper conducted in 2018. The former model was trained and saved in old Tensorflow format that only contains the weights with no information of its architecture graph--the frozen graph. This form of model could not be saved into a newer Tensorflow format like `.h5` or `.keras`, which also limiting the possibility to convert it to another framework like `onnx`. The end goal for the project is to enable Marco model deployment in local platforms.

This repository simply reconstruct the Marco model from scratch--layer by layer--referring to its paper and standard [InceptionV3 implementation in Keras API](https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/inception_v3.py) and [Tensorflow Slim](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py).

## Environment Installation
### Python Virtual Environment
This project requires Python 3.9.23 or higher.
```bash
$ python -m venv marco-rebuild
$ source marco-rebuild/bin/activate
$ pip install -r requirements.txt
```
### Conda Environment
```bash
$ conda create -n marco-rebuild python=3.9.23
$ conda activate marco-rebuild
$ pip install -r requirements.txt
```
## Usage

```bash
python main.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.