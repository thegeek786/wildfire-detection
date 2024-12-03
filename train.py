import torch
import torch.nn as nn
import torchvision.models as models
from dataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import datetime
import os
import random
from models import Logistic_two_stream,Flame_one_stream,VGG16,Vgg_two_stream,Logistic,Flame_two_stream,Mobilenetv2,Mobilenetv2_two_stream,LeNet5_one_stream,LeNet5_two_stream,Resnet18,Resnet18_two_stream
import json
import argparse
from torchmetrics.functional import precision, recall, f1_score
import csv
import numpy as np
import sys

def test(dataloader,net,DEVICE,MODE,Model_name,name_index,log_path,correct_on_test,correct_on_train,Model_custom_list, flag='test_set',output_flag = False):
    correct = 0
    total = 0    
    y_all_true = torch.tensor([]).to('cuda')
    y_all_pre  = torch.tensor([]).to('cuda')
    with torch.no_grad():
        net.eval()
        for  (rgb, ir, y)  in dataloader:
            y = y.to(DEVICE)
            y_all_true = torch.cat((y_all_true,y))
                        
            if Model_name in Model_custom_list:
                y_pre = net(rgb.to(DEVICE),ir.to(DEVICE),mode = MODE)
            else:
                if MODE=='rgb':
                    x = rgb.to(DEVICE)
                elif  MODE=='ir':
                    x = ir.to(DEVICE)
                    
                y_pre = net(x)
                        
            _, label_index = torch.max(y_pre.data, dim=-1)
            y_all_pre = torch.cat((y_all_pre,label_index))
            
            total += label_index.shape[0]
            correct += (label_index == y).sum().item()
            
        if flag == 'test_set':
            
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))
        
    if output_flag == True:
        macro_f1 = f1_score(y_all_pre.int(), y_all_true.int(), num_classes=3,average ='macro', task='multiclass')
        micro_f1 = f1_score(y_all_pre.int(), y_all_true.int(), num_classes=3, average ='micro',task='multiclass')
        
        with open(log_path,'a+',newline='') as f:
            csv_write = csv.writer(f)
            data_row = [ Model_name,MODE,name_index,macro_f1.item() ,micro_f1.item()]
            csv_write.writerow(data_row)
        
    return correct / total
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run dgr experiment.')            
    parser.add_argument('--path_rgb', type=str, default='254p RGB Images', help='results path') 
    parser.add_argument('--path_ir', type=str, default='254p Thermal Images', help='results path')
    parser.add_argument('--index', type=int, default= 0, help='name index. any number is fine')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, 1e-4 for small subset') 
    parser.add_argument('--classes_num', type=int, default=3, help='class number')
    parser.add_argument('--subset_rate', type=float, default=0.01, help='The rate of subset')
    parser.add_argument('--trainset_rate', type=float, default=0.8, help='split data to training and test')
    parser.add_argument('--model', type=str, default='Mobilenetv2_two_stream', help='VGG16 and Flame_one_stream')
    parser.add_argument('--mode', type=str, default='both', help='rgb/ir/both')        
    parser.add_argument('--EPOCH', type=int, default=50, help='Epoch for training')
    parser.add_argument('--test_interval', type=int, default=1, help='interval to report the results')
    parser.add_argument('--log_path', type=str, default='code/log/results.csv', help='results path')
    parser.add_argument('--log_loss_path', type=str, default='code/log/', help='results path to store loss info')
    args = parser.parse_args()
    print(args)
    log_path =  args.log_path

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f'use device: {DEVICE}')

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    classes_num = args.classes_num 
    BATCH_SIZE = args.batch_size
    LR = args.lr
    test_interval = args.test_interval
    EPOCH =args.EPOCH
    MODE = args.mode  # 'only rgb' 
    Model_name  = args.model
    name_index = args.index
    
    Transform_flag = False # sometime use it to resize the image
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f'use device: {DEVICE}')
        
    file_name =Model_name+'_'+MODE+'_'+str(args.index)
    Model_custom_list = ['Mobilenetv2_two_stream']
   
    if Model_name == 'Mobilenetv2_two_stream':
        net = Mobilenetv2_two_stream().to(DEVICE)
        Transform_flag = True
        target_size = 224
        
          
    print(net)
    path_rgb = args.path_rgb
    path_ir = args.path_ir
    
    Dataset = MyDataset(path_rgb, path_ir,input_size=target_size,transform=Transform_flag)            
    Dataset,_ = torch.utils.data.random_split(Dataset, [int(len(Dataset)*args.subset_rate), len(Dataset)-int(len(Dataset)*args.subset_rate)])
    split_rate = args.trainset_rate
    train_set, val_set = torch.utils.data.random_split(Dataset, [int(len(Dataset)*split_rate), len(Dataset)-int(len(Dataset)*split_rate)])
    train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=val_set, batch_size=16, shuffle=False)
    optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=0.00)
    loss_function =nn.CrossEntropyLoss(label_smoothing=0.2)
    correct_on_train = [0]
    correct_on_test = [0]
    total_step = (int(len(Dataset)*split_rate)// BATCH_SIZE  )        
    net.train()
    max_accuracy = 0
    Loss_accuracy = []
    test_acc = []
    loss_list = []
    train_acc = []    
    for index in range(EPOCH):
        net.train()
        for i, (rgb, ir, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            if Model_name in Model_custom_list:
                y_pre = net(rgb.to(DEVICE),ir.to(DEVICE),mode = MODE)
            else:
                if MODE=='rgb':
                    x = rgb.to(DEVICE)
                elif  MODE=='ir':
                    x = ir.to(DEVICE)
                    
                y_pre = net(x) 
            loss = loss_function(y_pre, y.to(DEVICE))
            Loss_accuracy.append(loss)
            print(f'Epoch:{index + 1}/{EPOCH}     Step:{i+1}|{total_step}   loss:{loss.item()}  ')
            loss_list.append(loss.item())
    
            loss.backward()
    
            optimizer.step()

        if ((index + 1) % test_interval) == 0:
            current_accuracy =test(test_dataloader,net,DEVICE,MODE,Model_name,name_index,log_path,correct_on_test,correct_on_train,Model_custom_list, flag='test_set',output_flag = False)
            test_acc.append(current_accuracy)
            current_accuracy_train = test(train_dataloader,net,DEVICE,MODE,Model_name,name_index,log_path,correct_on_test,correct_on_train,Model_custom_list, flag='train_set',output_flag = False)
            train_acc.append(current_accuracy_train)
            print(f'current max accuracy\t test set:{max(correct_on_test)}%\t train set:{max(correct_on_train)}%')
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
    
    test(test_dataloader,net,DEVICE,MODE,Model_name,name_index,log_path,correct_on_test,correct_on_train,Model_custom_list, flag='test_set',output_flag = True)                
    results_array = np.zeros(3,dtype=object)
    results_array[0]= loss_list
    results_array[1]= train_acc
    results_array[2]= test_acc    
    args.log_loss_path+file_name+'.npy'    
    np.save(args.log_loss_path+file_name+'.npy',results_array)
    sys.exit()