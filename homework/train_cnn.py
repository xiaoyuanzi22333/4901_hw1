from models import CNNClassifier, save_model, SoftmaxCrossEntropyLoss
from utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
import torchvision
import torch.utils.tensorboard as tb
from tqdm import tqdm

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    """
    Your code here
    """
    datapath = '../train_set'
    dataset = load_data(dataset_path = datapath)
    loss_fn = SoftmaxCrossEntropyLoss
    optimizer = optim.Adam(model.parameters(),lr = 1e-3, betas = (0.5, 0.999))
    num_epochs = 10
    
    for epoch in num_epochs:
        print("it is at epoch: ", epoch)
        for x,label in tqdm(dataset):
            if len(x) != batch_size:
                continue
            scores = model(x)
            loss = loss_fn(scores,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
