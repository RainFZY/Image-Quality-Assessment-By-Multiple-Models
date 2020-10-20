## Image Quality Assessment by Mutiple Models

### Prepare

1. download the database you need and put it under ./data, e.g. ./data/LIVE/fastfading ...

   LIVE (release2 recommended): http://live.ece.utexas.edu/research/Quality/subjective.htm

   tid2008: http://www.ponomarenko.info/tid2008.htm

   tid2013: http://www.ponomarenko.info/tid2013.htm

2. specify `datainfo` and`im_dir` in `config.yaml`



### Training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model='resnet18' --database='LIVE'
```

- Train/Val/Test split ratio in intra-database experiments can be set in `config.yaml` (default is 0.6/0.2/0.2).


- Compare different models' performance:


![](https://github.com/RainFZY/Image-Quality-Assessment-By-Multiple-Models/blob/master/data/images/compare.jpg)



### Test Demo

*Input an image and output its IQA score*

run:

```bash
python test_demo.py --im_path=data/images/test_images/blur.jpg --model_file=models/resnet18-LIVE
```

- --im_path: put your test image in the folder -- data/test_images

- --model_file: choose your trained model, the pre-trained resnet18-LIVE model is given



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

- *localhost: localhost's IP address*
- *user: user's name in host*
- *host: host's IP address*

**See the visualization in your PC:**

Enter localhost:16006 in the browser

![](https://github.com/RainFZY/Image-Quality-Assessment-By-Multiple-Models/blob/master/data/images/tensorboard.png)



### Application

See [IQA Distortion Classification and Reconstruction System](https://github.com/RainFZY/IQA-Distortion-Classification-and-Reconstruction-System)



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

