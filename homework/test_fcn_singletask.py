import torch
import numpy as np

from models import FCN_ST 
from utils import load_dense_data, ConfusionMatrix
import dense_transforms
import torch.utils.tensorboard as tb
from tqdm import tqdm



def CrossEntropyLoss_func():
    return torch.nn.CrossEntropyLoss()


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your single-task model, and perform evaluation for the segmentation task
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task
     
    """
    model = FCN_ST()
    
    caculator = ConfusionMatrix(19)
    
    cuda_device = 2
    model_path = './modelforp4/final_save.pth'
    dataset_path = './part4/test'
    batch_size = 32
    count = 0
    acc = 0
    
    dataset = load_dense_data(dataset_path = dataset_path,batch_size = batch_size)
    model = torch.load(model_path)
    model.eval()
    model = model.cuda(cuda_device)
    
    for x,label,depth_GT in tqdm(dataset):
        x = x.type(torch.FloatTensor)
        x = x.cuda(cuda_device)
        label = label.type(torch.FloatTensor)
        label = label.cuda(cuda_device)
        
        
        if len(x) != batch_size:
            continue
        # print(x.shape)
        scores = model(x)
        caculator.add(scores.argmax(1),label)
        
    mIoU = caculator.iou
    accuracy = caculator.global_accuracy
    print(mIoU)
    print("the accuracy is: ", caculator.global_accuracy)


    return accuracy, mIoU


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
