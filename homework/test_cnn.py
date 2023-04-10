from models import CNNClassifier
from utils import ConfusionMatrix, load_data
import torch
import torchvision
import torch.utils.tensorboard as tb
from tqdm import tqdm

def test(args):
    from os import path
    model = CNNClassifier()

    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for the vehicle classification task
    Hint: use the ConfusionMatrix for you to calculate accuracy
    """
    caculator = ConfusionMatrix(6)
    
    cuda_device = 2
    model_path = './modelforp3/epoch29_save.pth'
    dataset_path = './validation_subset'
    batch_size = 32
    count = 0
    acc = 0
    
    dataset = load_data(dataset_path = dataset_path,batch_size = batch_size)
    model = torch.load(model_path)
    model.eval()
    model = model.cuda(cuda_device)
    
    for x,label in tqdm(dataset):
        x = x.cuda(cuda_device)
        label = label.cuda(cuda_device)
        
        if len(x) != batch_size:
            continue
        # print(x.shape)
        scores = model(x)
        caculator.add(scores.argmax(1),label)
        # base += 1
    
    accuracy = caculator.global_accuracy
    print("the accuracy is: ", caculator.global_accuracy)
    
    return accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
