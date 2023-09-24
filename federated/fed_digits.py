"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
from nets.models import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils

def prepare_data(args):
    # Prepare data

    # Caltech
    Caltech_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Caltech/")
    Caltech_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Caltech/")
    Caltech_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Caltech/")
    Caltech_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Caltech/")
    # KKI
    KKI_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/KKI/")
    KKI_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/KKI/")
    KKI_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/KKI/")
    KKI_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/KKI/")
    #CMU
    CMU_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/CMU/")
    CMU_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/CMU/")
    CMU_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/CMU/")
    CMU_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/CMU/")
    #Leuven
    Leuven_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Leuven/")
    Leuven_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Leuven/")
    Leuven_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Leuven/")
    Leuven_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Leuven/")
    #NYU
    NYU_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/NYU/")
    NYU_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/NYU/")
    NYU_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/NYU/")
    NYU_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/NYU/")
    #MaxMun
    MaxMun_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/MaxMun/")
    MaxMun_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/MaxMun/")
    MaxMun_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/MaxMun/")
    MaxMun_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/MaxMun/")
    #OHSU
    OHSU_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/OHSU/")
    OHSU_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/OHSU/")
    OHSU_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/OHSU/")
    OHSU_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/OHSU/")
    #Olin
    Olin_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Olin/")
    Olin_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Olin/")
    Olin_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Olin/")
    Olin_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Olin/")
    #Pitt
    Pitt_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Pitt/")
    Pitt_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Pitt/")
    Pitt_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Pitt/")
    Pitt_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Pitt/")
    #SBL
    SBL_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/SBL/")
    SBL_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/SBL/")
    SBL_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/SBL/")
    SBL_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/SBL/")
    #SDSU
    SDSU_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/SDSU/")
    SDSU_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/SDSU/")
    SDSU_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/SDSU/")
    SDSU_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/SDSU/")
    #Stanford
    Stanford_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Stanford/")
    Stanford_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Stanford/")
    Stanford_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Stanford/")
    Stanford_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Stanford/")
    #Trinity
    Trinity_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Trinity/")
    Trinity_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Trinity/")
    Trinity_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Trinity/")
    Trinity_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Trinity/")
    #UCLA
    UCLA_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/UCLA/")
    UCLA_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/UCLA/")
    UCLA_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/UCLA/")
    UCLA_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/UCLA/")
    #UM
    UM_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/UM/")
    UM_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/UM/")
    UM_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/UM/")
    UM_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/UM/")
    #USM
    USM_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/USM/")
    USM_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/USM/")
    USM_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/USM/")
    USM_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/USM/")
    #Yale
    Yale_trainset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Yale/")
    Yale_testset_x1 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_200/Yale/")
    Yale_trainset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Yale/")
    Yale_testset_x2 = data_utils.DigitsDataset(data_path="../../FedBN-master/data/FL_392/Yale/")
    ############
    Caltech_train_loader_x1 = torch.utils.data.DataLoader(Caltech_trainset_x1, batch_size=args.batch, shuffle=True)
    Caltech_test_loader_x1  = torch.utils.data.DataLoader(Caltech_testset_x1, batch_size=args.batch, shuffle=False)
    Caltech_train_loader_x2 = torch.utils.data.DataLoader(Caltech_trainset_x2, batch_size=args.batch, shuffle=True)
    Caltech_test_loader_x2  = torch.utils.data.DataLoader(Caltech_testset_x2, batch_size=args.batch, shuffle=False)

    KKI_train_loader_x1 = torch.utils.data.DataLoader(KKI_trainset_x1, batch_size=args.batch, shuffle=True)
    KKI_test_loader_x1  = torch.utils.data.DataLoader(KKI_testset_x1, batch_size=args.batch, shuffle=False)
    KKI_train_loader_x2 = torch.utils.data.DataLoader(KKI_trainset_x2, batch_size=args.batch, shuffle=True)
    KKI_test_loader_x2  = torch.utils.data.DataLoader(KKI_testset_x2, batch_size=args.batch, shuffle=False)

    CMU_train_loader_x1 = torch.utils.data.DataLoader(CMU_trainset_x1, batch_size=args.batch, shuffle=True)
    CMU_test_loader_x1  = torch.utils.data.DataLoader(CMU_testset_x1, batch_size=args.batch, shuffle=False)
    CMU_train_loader_x2 = torch.utils.data.DataLoader(CMU_trainset_x2, batch_size=args.batch, shuffle=True)
    CMU_test_loader_x2  = torch.utils.data.DataLoader(CMU_testset_x2, batch_size=args.batch, shuffle=False)

    Leuven_train_loader_x1 = torch.utils.data.DataLoader(Leuven_trainset_x1, batch_size=args.batch, shuffle=True)
    Leuven_test_loader_x1  = torch.utils.data.DataLoader(Leuven_testset_x1, batch_size=args.batch, shuffle=False)
    Leuven_train_loader_x2 = torch.utils.data.DataLoader(Leuven_trainset_x2, batch_size=args.batch, shuffle=True)
    Leuven_test_loader_x2  = torch.utils.data.DataLoader(Leuven_testset_x2, batch_size=args.batch, shuffle=False)

    NYU_train_loader_x1 = torch.utils.data.DataLoader(NYU_trainset_x1, batch_size=args.batch, shuffle=True)
    NYU_test_loader_x1  = torch.utils.data.DataLoader(NYU_testset_x1, batch_size=args.batch, shuffle=False)
    NYU_train_loader_x2 = torch.utils.data.DataLoader(NYU_trainset_x2, batch_size=args.batch, shuffle=True)
    NYU_test_loader_x2  = torch.utils.data.DataLoader(NYU_testset_x2, batch_size=args.batch, shuffle=False)

    MaxMun_train_loader_x1 = torch.utils.data.DataLoader(MaxMun_trainset_x1, batch_size=args.batch, shuffle=True)
    MaxMun_test_loader_x1  = torch.utils.data.DataLoader(MaxMun_testset_x1, batch_size=args.batch, shuffle=False)
    MaxMun_train_loader_x2 = torch.utils.data.DataLoader(MaxMun_trainset_x2, batch_size=args.batch, shuffle=True)
    MaxMun_test_loader_x2  = torch.utils.data.DataLoader(MaxMun_testset_x2, batch_size=args.batch, shuffle=False)

    OHSU_train_loader_x1 = torch.utils.data.DataLoader(OHSU_trainset_x1, batch_size=args.batch, shuffle=True)
    OHSU_test_loader_x1  = torch.utils.data.DataLoader(OHSU_testset_x1, batch_size=args.batch, shuffle=False)
    OHSU_train_loader_x2 = torch.utils.data.DataLoader(OHSU_trainset_x2, batch_size=args.batch, shuffle=True)
    OHSU_test_loader_x2  = torch.utils.data.DataLoader(OHSU_testset_x2, batch_size=args.batch, shuffle=False)

    Olin_train_loader_x1 = torch.utils.data.DataLoader(Olin_trainset_x1, batch_size=args.batch, shuffle=True)
    Olin_test_loader_x1  = torch.utils.data.DataLoader(Olin_testset_x1, batch_size=args.batch, shuffle=False)
    Olin_train_loader_x2 = torch.utils.data.DataLoader(Olin_trainset_x2, batch_size=args.batch, shuffle=True)
    Olin_test_loader_x2  = torch.utils.data.DataLoader(Olin_testset_x2, batch_size=args.batch, shuffle=False)

    Pitt_train_loader_x1 = torch.utils.data.DataLoader(Pitt_trainset_x1, batch_size=args.batch, shuffle=True)
    Pitt_test_loader_x1  = torch.utils.data.DataLoader(Pitt_testset_x1, batch_size=args.batch, shuffle=False)
    Pitt_train_loader_x2 = torch.utils.data.DataLoader(Pitt_trainset_x2, batch_size=args.batch, shuffle=True)
    Pitt_test_loader_x2  = torch.utils.data.DataLoader(Pitt_testset_x2, batch_size=args.batch, shuffle=False)

    SBL_train_loader_x1 = torch.utils.data.DataLoader(SBL_trainset_x1, batch_size=args.batch, shuffle=True)
    SBL_test_loader_x1  = torch.utils.data.DataLoader(SBL_testset_x1, batch_size=args.batch, shuffle=False)
    SBL_train_loader_x2 = torch.utils.data.DataLoader(SBL_trainset_x2, batch_size=args.batch, shuffle=True)
    SBL_test_loader_x2  = torch.utils.data.DataLoader(SBL_testset_x2, batch_size=args.batch, shuffle=False)

    SDSU_train_loader_x1 = torch.utils.data.DataLoader(SDSU_trainset_x1, batch_size=args.batch, shuffle=True)
    SDSU_test_loader_x1  = torch.utils.data.DataLoader(SDSU_testset_x1, batch_size=args.batch, shuffle=False)
    SDSU_train_loader_x2 = torch.utils.data.DataLoader(SDSU_trainset_x2, batch_size=args.batch, shuffle=True)
    SDSU_test_loader_x2  = torch.utils.data.DataLoader(SDSU_testset_x2, batch_size=args.batch, shuffle=False)

    Stanford_train_loader_x1 = torch.utils.data.DataLoader(Stanford_trainset_x1, batch_size=args.batch, shuffle=True)
    Stanford_test_loader_x1  = torch.utils.data.DataLoader(Stanford_testset_x1, batch_size=args.batch, shuffle=False)
    Stanford_train_loader_x2 = torch.utils.data.DataLoader(Stanford_trainset_x2, batch_size=args.batch, shuffle=True)
    Stanford_test_loader_x2  = torch.utils.data.DataLoader(Stanford_testset_x2, batch_size=args.batch, shuffle=False)

    Trinity_train_loader_x1 = torch.utils.data.DataLoader(Trinity_trainset_x1, batch_size=args.batch, shuffle=True)
    Trinity_test_loader_x1  = torch.utils.data.DataLoader(Trinity_testset_x1, batch_size=args.batch, shuffle=False)
    Trinity_train_loader_x2 = torch.utils.data.DataLoader(Trinity_trainset_x2, batch_size=args.batch, shuffle=True)
    Trinity_test_loader_x2  = torch.utils.data.DataLoader(Trinity_testset_x2, batch_size=args.batch, shuffle=False)

    UCLA_train_loader_x1 = torch.utils.data.DataLoader(UCLA_trainset_x1, batch_size=args.batch, shuffle=True)
    UCLA_test_loader_x1  = torch.utils.data.DataLoader(UCLA_testset_x1, batch_size=args.batch, shuffle=False)
    UCLA_train_loader_x2 = torch.utils.data.DataLoader(UCLA_trainset_x2, batch_size=args.batch, shuffle=True)
    UCLA_test_loader_x2  = torch.utils.data.DataLoader(UCLA_testset_x2, batch_size=args.batch, shuffle=False)
    
    UM_train_loader_x1 = torch.utils.data.DataLoader(UM_trainset_x1, batch_size=args.batch, shuffle=True)
    UM_test_loader_x1  = torch.utils.data.DataLoader(UM_testset_x1, batch_size=args.batch, shuffle=False)
    UM_train_loader_x2 = torch.utils.data.DataLoader(UM_trainset_x2, batch_size=args.batch, shuffle=True)
    UM_test_loader_x2  = torch.utils.data.DataLoader(UM_testset_x2, batch_size=args.batch, shuffle=False)

    USM_train_loader_x1 = torch.utils.data.DataLoader(USM_trainset_x1, batch_size=args.batch, shuffle=True)
    USM_test_loader_x1  = torch.utils.data.DataLoader(USM_testset_x1, batch_size=args.batch, shuffle=False)
    USM_train_loader_x2 = torch.utils.data.DataLoader(USM_trainset_x2, batch_size=args.batch, shuffle=True)
    USM_test_loader_x2  = torch.utils.data.DataLoader(USM_testset_x2, batch_size=args.batch, shuffle=False)

    Yale_train_loader_x1 = torch.utils.data.DataLoader(USM_trainset_x1, batch_size=args.batch, shuffle=True)
    Yale_test_loader_x1  = torch.utils.data.DataLoader(USM_testset_x1, batch_size=args.batch, shuffle=False)
    Yale_train_loader_x2 = torch.utils.data.DataLoader(USM_trainset_x2, batch_size=args.batch, shuffle=True)
    Yale_test_loader_x2  = torch.utils.data.DataLoader(USM_testset_x2, batch_size=args.batch, shuffle=False)


    train_loaders_x1 = [Caltech_train_loader_x1, KKI_train_loader_x1, CMU_train_loader_x1, Leuven_train_loader_x1, NYU_train_loader_x1, MaxMun_train_loader_x1, OHSU_train_loader_x1, Olin_train_loader_x1, Pitt_train_loader_x1,SBL_train_loader_x1,SDSU_train_loader_x1,Stanford_train_loader_x1, Trinity_train_loader_x1,UCLA_train_loader_x1,UM_train_loader_x1,USM_train_loader_x1, Yale_train_loader_x1]
    train_loaders_x2 = [Caltech_train_loader_x2, KKI_train_loader_x2, CMU_train_loader_x2, Leuven_train_loader_x2, NYU_train_loader_x2, MaxMun_train_loader_x2, OHSU_train_loader_x2, Olin_train_loader_x2, Pitt_train_loader_x2,SBL_train_loader_x2,SDSU_train_loader_x2,Stanford_train_loader_x2, Trinity_train_loader_x2,UCLA_train_loader_x2,UM_train_loader_x2,USM_train_loader_x2, Yale_train_loader_x2]
    test_loaders_x1 = [Caltech_test_loader_x1, KKI_test_loader_x1, CMU_test_loader_x1, Leuven_test_loader_x1, NYU_test_loader_x1, MaxMun_test_loader_x1, OHSU_test_loader_x1, Olin_test_loader_x1, Pitt_test_loader_x1,SBL_test_loader_x1,SDSU_test_loader_x1,Stanford_test_loader_x1, Trinity_test_loader_x1,UCLA_test_loader_x1,UM_test_loader_x1,USM_test_loader_x1, Yale_test_loader_x1]
    test_loaders_x2 = [Caltech_test_loader_x2, KKI_test_loader_x2, CMU_test_loader_x2, Leuven_test_loader_x2, NYU_test_loader_x2, MaxMun_test_loader_x2, OHSU_test_loader_x2, Olin_test_loader_x2, Pitt_test_loader_x2,SBL_test_loader_x2,SDSU_test_loader_x2,Stanford_test_loader_x2, Trinity_test_loader_x2,UCLA_test_loader_x2,UM_test_loader_x2,USM_test_loader_x2, Yale_test_loader_x2]

    return train_loaders_x1,train_loaders_x2, test_loaders_x1, test_loaders_x2 

def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter_x1 = iter(train_loader_x1)
    train_iter_x2 = iter(train_loader_x2)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x_1, y = next(train_iter_x1)
        x_2, y = next(train_iter_x2)
        num_data += y.size(0)
        x_1 = x_1.to(device).float()
        x_2 = x_2.to(device).float()
        y = y.to(device).long()
        output = model(x_1,x_2)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def train_fedprox(args, model, train_loader_x1,train_loader_x2 , optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter_x1 = iter(train_loader_x1)
    train_iter_x2 = iter(train_loader_x2)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x_1, y = next(train_iter_x1)
        x_2, y = next(train_iter_x2)
        num_data += y.size(0)
        x_1 = x_1.to(device).float()
        x_2 = x_2.to(device).float()
        y = y.to(device).long()
        output = model(x_1,x_2)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def test(model, test_loader_x1,test_loader_x2 , loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output = model(data)
        
        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
    
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 

    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedbn', help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint/digits', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    args = parser.parse_args()

    exp_folder = 'federated_digits'

    args.save_path = os.path.join(args.save_path, exp_folder)
    
    log = args.log
    if log:
        log_path = os.path.join('../logs/digits/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
        logfile.write('    wk_iters: {}\n'.format(args.wk_iters))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
   
   
    server_model = DigitModel().to(device)
    loss_fun = nn.CrossEntropyLoss()

    # prepare the data
    train_loaders_x1, test_loaders_x1 = prepare_data(args)
    train_loaders_x2, test_loaders_x2 = prepare_data(args)

    # name of each client dataset
    datasets = ['Caltech','KKI', 'CMU', 'Leuven', 'NYU', 'MaxMun', 'OHSU', 'Olin', 'Pitt', 'SBL', 'SDSU', 'Stanford','Trinity', 'UCLA', 'UM', 'USM', 'Yale']
    
    # federated setting
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../home/amahdizadeh/FedBN-master/snapshots/digits/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        else:
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(server_model, test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0

    # start training
    for a_iter in range(resume_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log: logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 
            
            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device)
                else:
                    train(model, train_loader, optimizer, loss_fun, client_num, device)
         
        # aggregation
        server_model, models = communication(args, server_model, models, client_weights)
        
        # report after aggregation
        for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train_loss, train_acc = test(model, train_loader, loss_fun, device) 
                print(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                if args.log:
                    logfile.write(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))\

        # start testing
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            if args.log:
                logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss, test_acc))

    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    if args.mode.lower() == 'fedbn':
        torch.save({
            'model_0': models[0].state_dict(),
            'model_1': models[1].state_dict(),
            'model_2': models[2].state_dict(),
            'model_3': models[3].state_dict(),
            'model_4': models[4].state_dict(),
            'model_5': models[4].state_dict(),
            'model_6': models[4].state_dict(),
            'model_7': models[4].state_dict(),
            'model_8': models[4].state_dict(),
            'model_9': models[4].state_dict(),
            'model_10': models[4].state_dict(),
            'model_11': models[4].state_dict(),
            'model_12': models[4].state_dict(),
            'model_13': models[4].state_dict(),
            'model_14': models[4].state_dict(),
            'model_15': models[4].state_dict(),
            'model_16': models[4].state_dict(),
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)
    else:
        torch.save({
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()


