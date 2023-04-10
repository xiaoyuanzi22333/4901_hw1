from models import CNNClassifier, save_model, SoftmaxCrossEntropyLoss
from utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
import torchvision
import torch.utils.tensorboard as tb
from tqdm import tqdm


def CrossEntropyLoss_func():
    return torch.nn.CrossEntropyLoss()


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
    cuda_device = 1
    batch_size = 64
    
    datapath = './train_subset'
    modelpath = './modelforp3'
    
    
    dataset = load_data(dataset_path = datapath,batch_size = batch_size)
    
    # loss_fn = SoftmaxCrossEntropyLoss()
    loss_fn = CrossEntropyLoss_func()
    
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3, betas = (0.5, 0.999))
    num_epochs = 30
    
    model = model.cuda(cuda_device)
    # loss_fn = loss_fn.cuda(cuda_device)
    # optimizer = optimizer.cuda(cuda_device)
    
    print("model's device is: ", next(model.parameters()).device)
    logger = tb.SummaryWriter('cnn')
    
    for epoch in range(num_epochs):
        loss = 0.0
        count = 0
        base = 0
        for x,label in tqdm(dataset):
            x = x.cuda(cuda_device)
            label = label.cuda(cuda_device)
            
            # print(x.shape)
            
            if len(x) != batch_size:
                continue
            # print(x.shape)
            scores = model(x)
            
            
            # print(scores.shape)
            # print(label.shape)
            
            loss = loss_fn(scores,label)
            # print(loss)
            for i in range(batch_size):
                if max(scores[i]) == scores[i][label[i]]:
                    count += 1
                base += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1)%10 == 0:
            torch.save(model,modelpath+'/epoch'+str(epoch)+'_save.pth')
        
        accuracy = count/base
        print("epoch: ",epoch+1,", train loss: ",loss, ", accuracy: ",accuracy)
        logger.add_scalar('loss',loss,epoch)
        logger.add_scalar('accuracy',accuracy,epoch)
    
    torch.save(model,model_path+'/final_save.pth')
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
