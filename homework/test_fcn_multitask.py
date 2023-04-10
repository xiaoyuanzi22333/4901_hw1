import torch
import numpy as np

from models import FCN_MT 
from utils import load_dense_data, ConfusionMatrix, DepthError
import dense_transforms
import torch.utils.tensorboard as tb
from tqdm import tqdm


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for both segmentation and depth estimation tasks
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task
    Hint: use DepthError for you to calculate rel, a1, a2, and a3 for the depth estimation task. 
    """
    
    caculator = ConfusionMatrix(19)
    
    cuda_device = 0
    model_path = './modelforp4(2)/final_save.pth'
    dataset_path = './part4/val'
    batch_size = 32
    count = 0
    acc = 0
    rel = 0.0
    base = 0
    abs= 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    
    dataset = load_dense_data(dataset_path = dataset_path,batch_size = batch_size)
    model = torch.load(model_path)
    model.eval()
    model = model.cuda(cuda_device)
    
    for x,label,depth_GT in tqdm(dataset):
        x = x.type(torch.FloatTensor)
        x = x.cuda(cuda_device)
        label = label.type(torch.FloatTensor)
        label = label.cuda(cuda_device)
        depth_GT = depth_GT.type(torch.FloatTensor)
        depth_GT = depth_GT.cuda(cuda_device)
        
        
        if len(x) != batch_size:
            continue
        # print(x.shape)
        scores, depth = model(x)
        caculator.add(scores.argmax(1),label)
        depth = depth.cpu().detach().numpy()
        depth_GT = depth_GT.cpu().detach().numpy()
        abs, error1, error2, error3 = DepthError(depth,depth_GT).compute_errors
        rel += abs
        a1 += error1
        a2 += error2
        a3 += error3
        base += 1
    
    accuracy = caculator.global_accuracy
    mIoU = caculator.iou
    rel = rel/base
    a1 = a1/base
    a2 = a2/base
    a3 = a3/base
    print(a1)
    print(a2)
    print(a3)
    print(rel)
    print(mIoU)
    print("the accuracy is: ", caculator.global_accuracy)
    

    return accuracy, mIoU, rel, a1, a2, a3


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
