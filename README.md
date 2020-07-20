## Image Quality Assessment by Mutiple Models

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model='resnet18' --database='LIVE'
```

Before training, the `im_dir` in `config.yaml` must to be specified.
Train/Val/Test split ratio in intra-database experiments can be set in `config.yaml` (default is 0.6/0.2/0.2).

### Evaluation

Test Demo

```bash
python test_demo.py --im_path=data/I03_01_1.bmp
```

### Cross Dataset

```bash
python test_cross_dataset.py --help
```

TODO: add metrics calculation. SROCC, KROCC can be easily get. PLCC, RMSE, MAE, OR should be calculated after a non-linear fitting since the quality score ranges are not the same across different IQA datasets.

### Visualization

```bash
tensorboard --logdir=tensorboard_logs --port=6006 # in the server (host:port)
ssh -p port -L 6006:localhost:6006 user@host # in your PC. See the visualization in your PC
```

### Requirements

```bash
conda create -n reproducibleresearch pip python=3.6
source activate reproducibleresearch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
source deactive
```

- Python 3.6.8
- PyTorch 1.3.0
- TensorboardX 1.9, TensorFlow 2.0.0
- [pytorch/ignite 0.2.1](https://github.com/pytorch/ignite)

Note: You need to install the right CUDA version.

