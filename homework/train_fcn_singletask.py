import torch
import numpy as np
from models import FCN_ST, save_model
from utils import ConfusionMatrix , load_dense_data
# from utils import *
import dense_transforms
import torch.utils.tensorboard as tb
from tqdm import tqdm 
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"




def CrossEntropyLoss_func(loss_weights):
    return torch.nn.CrossEntropyLoss(loss_weights, ignore_index=-1)



def train(args):
    from os import path
    model = FCN_ST()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: Use dense_transforms for data augmentation. If you found a good data augmentation parameters for the CNN, use them here too.
    Hint: Use the log function below to debug and visualize your model
    """
    
    model = FCN_ST()
    
    cuda_device = 4
    batch_size = 16
    
    datapath = './part4/train'
    modelpath = './modelforp4'
    
    loss_weights = torch.from_numpy(np.array([3.29,21.9,4.68,121.32,266.84,117.6,1022.23,205.68,
              6.13,118.81,35.17,168.36,460.62,15.53,272.62,501.94,3536.12,2287.91,140.32]))
    loss_weights = loss_weights.type(torch.FloatTensor)
    loss_weights = loss_weights.cuda(cuda_device)
    
    
    dataset = load_dense_data(dataset_path = datapath,batch_size = batch_size)
    caculator = ConfusionMatrix(19)

    loss_fn = CrossEntropyLoss_func(loss_weights)
    
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3, betas = (0.5, 0.999))
    num_epochs = 300
    
    # model = model.float()
    model = model.cuda(cuda_device)

    print("model's device is: ", next(model.parameters()).device)
    logger = tb.SummaryWriter('test_fcn_st')
    
    for epoch in range(num_epochs):
        for x,label,depth in tqdm(dataset):
            x = x.type(torch.FloatTensor)
            x = x.cuda(cuda_device)
            label = label.type(torch.LongTensor)
            label = label.cuda(cuda_device)
            label = label.squeeze()
            # print(torch.min(label))
            
            # print(x.dtype)
            # print(label.dtype)
            
            
            # print(x.shape)
            
            if len(x) != batch_size:
                continue
            # print(x.shape)
            scores = model(x)
            scores = scores.type(torch.FloatTensor)
            scores = scores.cuda(cuda_device)
            # print(scores.shape)
            # print(label.shape)
            loss = loss_fn(scores,label)
            caculator.add(scores.argmax(1),label)
            # print(loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1)%10 == 0:
            torch.save(model,modelpath+'/epoch'+str(epoch)+'_save.pth')
        
        print("epoch: ",epoch+1,", train loss: ",loss, ", accuracy: ",caculator.global_accuracy,', miou: ', caculator.iou)
        logger.add_scalar('loss',loss,epoch)
        logger.add_scalar('accuracy',caculator.global_accuracy,epoch)
        logger.add_scalar('miou', caculator.iou,epoch)
    
    torch.save(model,modelpath+'/final_save.pth')
    
    
    # save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
