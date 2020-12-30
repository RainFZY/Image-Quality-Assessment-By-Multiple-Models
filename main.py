"""
PyTorch 1.3 implementation of the following paper:
Kang L, Ye P, Li Y, et al. Convolutional neural networks for no-reference image quality assessment[C]//
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1733-1740.
 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=logger --port=6006
    ```
    Run the main.py:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --exp_id=0
    ```

 Implemented by Zhuoyu Feng
 Email: dingquanli@pku.edu.cn
 Date: 2019/11/8
"""

from argparse import ArgumentParser
import os
import numpy as np
import random
from scipy import stats
import yaml
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from IQADataset import IQADataset
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric
from tensorboardX import SummaryWriter
import datetime
import functools
from models import resnet, vgg, cnn, lenet5
from torchsummary import summary
from torch.autograd import Variable

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 平均绝对误差（MAE），目标变量和预测变量之间绝对差值之和
def loss_fn(y_pred, y):
    return F.l1_loss(y_pred, y[0])


class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._y_pred = []
        self._y      = []
        self._y_std  = []

    def update(self, output):
        y_pred, y = output

        self._y.append(y[0].item())
        self._y_std.append(y[1].item())
        self._y_pred.append(torch.mean(y_pred).item())

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,)) # label
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,)) # prediction

        srocc = stats.spearmanr(sq, q)[0] # 计算两个输入的spearman相关系数
        krocc = stats.stats.kendalltau(sq, q)[0] # 肯德尔秩次相关系数
        plcc = stats.pearsonr(sq, q)[0] # 皮尔逊相关系数
        rmse = np.sqrt(((sq - q) ** 2).mean()) # 均方根误差
        mae = np.abs((sq - q)).mean() # 均方误差
        outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean() # 离出率

        return srocc, krocc, plcc, rmse, mae, outlier_ratio




def get_data_loaders(config, train_batch_size, exp_id=0):
    train_dataset = IQADataset(config, exp_id, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4)

    val_dataset = IQADataset(config, exp_id, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = IQADataset(config, exp_id, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset)

        return train_loader, val_loader, test_loader

    return train_loader, val_loader


def run(train_batch_size, epochs, lr, weight_decay, config, exp_id, log_dir, trained_model_file, model_name, disable_gpu=False):
    # 日志工具
    def logging(s, log_path, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(log_path, 'a+') as f_log:
                f_log.write(s + '\n')
    def get_logger(log_path, **kwargs):
        return functools.partial(logging, log_path=log_path, **kwargs)

    logging = get_logger('./logger/log.txt')

    # 加载数据集
    if config['test_ratio']:
        train_loader, val_loader, test_loader = get_data_loaders(config, train_batch_size, exp_id)
    else:
        train_loader, val_loader = get_data_loaders(config, train_batch_size, exp_id)

    device = torch.device("cuda" if not disable_gpu and torch.cuda.is_available() else "cpu")

    if model_name == 'CNNIQA':
        model = cnn.CNNIQAnet()
    if model_name == 'lenet5':
        model = lenet5.LeNet5()
    if model_name == 'resnet18':
        model = resnet.ResNet18()
    if model_name == 'resnet34':
        model = resnet.ResNet34()
    if model_name == 'vgg19':
        model = vgg.VGG('VGG19')

    writer = SummaryWriter(log_dir=log_dir)
    model = model.to(device) # 将模型加载到指定设备上
    # summary(model, input_size=(32, 32))  # must remove the number of N

    # print("model:", model)
    # logging("model: {}".format(model))
    # if multi_gpu and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    global best_criterion
    best_criterion = -1  # SROCC >= -1
    # 训练器，调库
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    # 校验器，调库
    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance()},
                                            device=device)

    # 函数修饰器，以下函数都包含在trainer中，因此是一边训练一边验证、测试
    # training/validation/testing = 0.6/0.2/0.2，每一个epoch训练完都进行validation和testing
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
        # print("Validation Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
        #       .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
        logging("Validation Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
              .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
        writer.add_scalar("validation/SROCC", SROCC, engine.state.epoch)
        writer.add_scalar("validation/KROCC", KROCC, engine.state.epoch)
        writer.add_scalar("validation/PLCC", PLCC, engine.state.epoch)
        writer.add_scalar("validation/RMSE", RMSE, engine.state.epoch)
        writer.add_scalar("validation/MAE", MAE, engine.state.epoch)
        writer.add_scalar("validation/OR", OR, engine.state.epoch)
        global best_criterion
        global best_epoch
        if SROCC > best_criterion:
            best_criterion = SROCC
            best_epoch = engine.state.epoch
            # 保存最佳模型，以SROCC指标为准
            # _use_new_zipfile_serialization = False适用于pytorch1.6以前的版本
            torch.save(model.state_dict(), trained_model_file, _use_new_zipfile_serialization=False)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(engine):
        if config["test_ratio"] > 0 and config['test_during_training']:
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            # print("Testing Results    - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
            #       .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            logging("Testing Results     - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                  .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            writer.add_scalar("testing/SROCC", SROCC, engine.state.epoch)
            writer.add_scalar("testing/KROCC", KROCC, engine.state.epoch)
            writer.add_scalar("testing/PLCC", PLCC, engine.state.epoch)
            writer.add_scalar("testing/RMSE", RMSE, engine.state.epoch)
            writer.add_scalar("testing/MAE", MAE, engine.state.epoch)
            writer.add_scalar("testing/OR", OR, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        if config["test_ratio"]:
            model.load_state_dict(torch.load(trained_model_file))
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            global best_epoch
            # best test results 是 validation的SROCC最高的一次
            # print("Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
            #     .format(best_epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            logging("Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                .format(best_epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            np.save(save_result_file, (SROCC, KROCC, PLCC, RMSE, MAE, OR))

    # kick everything off
    # 执行训练
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == "__main__":
    def logging(s, log_path, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(log_path, 'a+') as f_log:
                f_log.write(s + '\n')
    def get_logger(log_path, **kwargs):
        return functools.partial(logging, log_path=log_path, **kwargs)

    parser = ArgumentParser(description='PyTorch CNNIQA')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='LIVE', type=str,
                        help='database name (default: LIVE)')
    parser.add_argument('--model', default='CNNIQA', type=str,
                        help='model name (default: CNNIQA)')
    # parser.add_argument('--resume', default=None, type=str,
    #                     help='path to latest checkpoint (default: None)')
    parser.add_argument("--log_dir", type=str, default="logger",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    # parser.add_argument('--multi_gpu', action='store_true',
    #                     help='flag whether to use multiple GPUs')

    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # print('exp id: ' + args.exp_id)
    # print('database: ' + args.database)
    # print('model: ' + args.model)
    logging = get_logger('./logger/log_'+ args.model + '_' + args.database + '.txt')
    logging('exp id: {}'.format(args.exp_id))
    logging("database: {}".format(args.database))
    logging("model: {}".format(args.model))

    config.update(config[args.database])
    # config.update(config[args.model])

    # log_dir = '{}/EXP{}-{}-{}-lr={}-{}'.format(args.log_dir, args.exp_id, args.database, args.model, args.lr,
    #                                               datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    log_dir = 'logger/test_log'

    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-epoch{}-lr={}'.format(args.model, args.database, args.epochs, args.lr)

    # main process
    run(args.batch_size, args.epochs, args.lr, args.weight_decay, config, args.exp_id,
        log_dir, trained_model_file, args.model, args.disable_gpu)
