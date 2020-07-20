## Image Quality Assessment by Mutiple Models

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model='resnet18' --database='LIVE'
```

Before training, the `im_dir` in `config.yaml` must to be specified.
Train/Val/Test split ratio in intra-database experiments can be set in `config.yaml` (default is 0.6/0.2/0.2).

Compare different models' performance:

![](https://github.com/RainFZY/Image-Quality-Assessment-By-Multiple-Models/blob/master/data/images/compare.jpg)

### Test Demo

put your test image in the folder -- data/test_images

choose your trained model, the pre-trained resnet18-LIVE model is given

```bash
python test_demo.py --im_path=data/images/test_images/blur.jpg --model_file=models/resnet18-LIVE
```

### Visualization

**In the server (host:port):**

```bash
tensorboard --logdir=tensorboard_logs --port=6006
```

e.g. put the dpai-11 file in logger/test_log, run:

```
tensorboard --logdir="./logger/test_log" --port=6006
```

**In your PC:**

```
ssh -p port -L 6006:localhost:6006 user@host
```

*localhost: localhost's IP address*

*user: user's name in host*

*host: host's IP address*

**See the visualization in your PC:**

Enter localhost:16006 in the browser

![](https://github.com/RainFZY/Image-Quality-Assessment-By-Multiple-Models/blob/master/data/images/tensorboard.png)

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

