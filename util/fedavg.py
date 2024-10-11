# python version 3.7.1
# -*- coding: utf-8 -*-

import copy
from pydoc import cli
import torch
import numpy as np
import multiprocessing

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        #print('k',k)
        for i in range(1, len(w)):
            #print('i',i)
            w_avg[k] += w[i][k]
            #print(w[i][k])
        #w_avg[k] = torch.div(w_avg[k], len(w))
        w_avg[k] = w_avg[k] / len(w)
    return w_avg


def diver_cal(model_flag, w_g, w_l):
    w_flag = model_flag.state_dict()
    for k in w_flag.keys():
        #print(k)
        w_flag[k] = w_g[k] - w_l[k]
    model_flag.load_state_dict(w_flag)
    sum_diver = 0
    for param in model_flag.parameters():
        #print('param',param)
        se = torch.sum(param**2)
        #print('se', se)
        sum_diver += se.detach().cpu().numpy()

    #sum_diver = s
    #print(s.detach().numpy())
    return sum_diver


def FedAvg_noniid(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg

def FedAvg_noniid_intervention(w, dict_len, grad=None, device=None):
    w_avg = copy.deepcopy(w[0])
    if device == None:
        w_grad_vec = torch.zeros(grad[0].shape)
    else:
        w_grad_vec = torch.zeros(grad[0].shape).to(device)
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * dict_len[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
            #w_avg[k] += w[i][k]
        #w_avg[k] = w_avg[k] / len(w)
        w_avg[k] = w_avg[k] / sum(dict_len)
    for i in range(len(grad)):
        w_grad_vec += grad[i] * dict_len[i] / sum(dict_len)

    return w_avg, w_grad_vec

def FedAvg_Rod(backbone_w_locals, dict_len, device):
    backbone_w_avg = FedAvg_noniid(backbone_w_locals, dict_len)
    return backbone_w_avg

def FedAvg_Rod_FedGrab(backbone_w_locals, linear_w_locals, dict_len, device, grad=None):
    backbone_w_avg = FedAvg_noniid(backbone_w_locals, dict_len)
    linear_w_avg = FedAvg_noniid(linear_w_locals, dict_len)
    if device == None:
        w_grad_vec = torch.zeros(grad[0].shape)
    else:
        w_grad_vec = torch.zeros(grad[0].shape).to(device)
    for i in range(len(grad)):
        w_grad_vec += grad[i] * dict_len[i] / sum(dict_len)
    return backbone_w_avg, linear_w_avg, w_grad_vec


def Communication_causal(w, grad_con):
    w_ca = copy.deepcopy(w[0])
    for name_param in w[0]:
        list_values_param = []
        for dict_local_params, c in zip(w, grad_con):
            list_values_param.append(dict_local_params[name_param] * c)
        value_global_param = sum(list_values_param)
        w_ca[name_param] = value_global_param
    return w_ca


def Communication_causal_1(w, grad_con, grad_vec, device=None):
    w_ca = copy.deepcopy(w[0])
    if device == None:
        w_grad_vec = torch.zeros(grad_vec[0].shape)
    else:
        w_grad_vec = torch.zeros(grad_vec[0].shape).to(device)
    for name_param in w[0]:
        list_values_param = []
        for dict_local_params, c in zip(w, grad_con):
            list_values_param.append(dict_local_params[name_param] * c)
        value_global_param = sum(list_values_param)
        w_ca[name_param] = value_global_param

    for i in range(len(grad_vec)):
        w_grad_vec += grad_vec[i] * grad_con[i]
    return w_ca,w_grad_vec

def FedAvg_Rod_intervention(backbone_w_locals, dict_len, grad_vec, device=None):
    backbone_w_avg, tail_grad = FedAvg_noniid_intervention(backbone_w_locals, dict_len, grad_vec,device)
    return backbone_w_avg,tail_grad



#==================FedGrab=================
def weno_aggeration(w, dict_len, datasetObj, beta, round, start_round=25):
    # fedavg的weight
    avg_w = copy.deepcopy(w[0])
    # 合并feature extractor(当然连同classifier一起合并了)
    for k in avg_w.keys():
        avg_w[k] = avg_w[k] * dict_len[0]
        for i in range(1, len(w)):
            avg_w[k] += w[i][k] * dict_len[i]
            # w_avg[k] += w[i][k]
        # w_avg[k] = w_avg[k] / len(w)
        avg_w[k] = avg_w[k] / sum(dict_len)
    # 计算weightnorm
    # wns_prop = []    # wns[i][j]:第i个client的classifier的第j类的client内占比（weight norms proportion）
    # for i in range(len(w)):
    #     wns_prop.append(torch.norm(w[i]["linear.weight"], p=2, dim=1) / torch.norm(w[i]["linear.weight"], p=2, dim=1).sum())

    # weno aggregation for classifier
    # weno_w["linear.weight"].zero_()
    # weno_w["linear.bias"].zero_()
    # class_wise_num = [0 for i in range(weno_w["linear.bias"].shape[0])]     # 每个类别的累计样本总数，十个类别则长度为十
    # for id_cls in range(weno_w["linear.bias"].shape[0]):   # 对于每一个类别而言
    #     for id_client in range(len(w)):   # 对于每一个client
    #         weno_w["linear.weight"][id_cls] += w[id_client]["linear.weight"][id_cls] * wns_prop[id_client][id_cls] * dict_len[id_cls]
    #         weno_w["linear.bias"][id_cls] += w[id_client]["linear.bias"][id_cls] * wns_prop[id_client][id_cls] * dict_len[id_cls]
    #         class_wise_num[id_cls] += wns_prop[id_client][id_cls] * dict_len[id_cls]
    #     weno_w["linear.weight"][id_cls] / class_wise_num[id_cls]
    #     weno_w["linear.bias"][id_cls] / class_wise_num[id_cls]

    # 完全weno
    weno_classifier = copy.deepcopy(w[0])
    client_distribution = datasetObj.training_set_distribution

    # 按照sample proportion进行聚合
    client_distribution = client_distribution.astype(np.float64)
    for i in range(len(client_distribution)):
        client_distribution[i] /= sum(client_distribution[i])
    weno_classifier["linear.weight"].zero_()
    weno_classifier["linear.bias"].zero_()
    class_wise_num = [0 for i in range(weno_classifier["linear.bias"].shape[0])]  # 每个类别的累计样本总数，十个类别则长度为十
    for id_cls in range(weno_classifier["linear.bias"].shape[0]):  # 对于每一个类别而言
        for id_client in range(len(w)):  # 对于每一个client
            weno_classifier["linear.weight"][id_cls] += w[id_client]["linear.weight"][id_cls] * \
                                                        client_distribution[id_client][id_cls] * dict_len[id_cls]
            weno_classifier["linear.bias"][id_cls] += w[id_client]["linear.bias"][id_cls] * \
                                                      client_distribution[id_client][id_cls] * dict_len[id_cls]
            class_wise_num[id_cls] += client_distribution[id_client][id_cls] * dict_len[id_cls]
        weno_classifier["linear.weight"][id_cls] / class_wise_num[id_cls]
        weno_classifier["linear.bias"][id_cls] / class_wise_num[id_cls]

    # 加权
    if round > start_round:
        avg_w["linear.weight"] = beta * weno_classifier["linear.weight"] + (1 - beta) * avg_w["linear.weight"]
        avg_w["linear.bias"] = beta * weno_classifier["linear.bias"] + (1 - beta) * avg_w["linear.bias"]

    return avg_w


def Weighted_avg_f1(f1_list,dict_len):
    f1_avg = 0
    for i in range(len(dict_len)):
        f1_avg += f1_list[i]*dict_len[i]
    f1_avg = f1_avg/sum(dict_len)
    return f1_avg