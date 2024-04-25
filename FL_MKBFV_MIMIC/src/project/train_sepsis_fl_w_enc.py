import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Subset
import torch.optim as optim

from utils import train, evaluate, best_evaluate
from mydatasets import calculate_num_features, VisitSequenceWithLabelDataset, time_collate_fn
from mymodels import MyLSTM
# from torchsummary import summary
import numpy as np
from enclib_forFL import *

sk_c1, sk_c2, sk_c3, sk_c4, pk = KeyGen()

# def SUM_w_RRAM_noise(para_plain_A , para_plain_B , para_plain_C , para_plain_D, quan_precision, RRAM_noise):
#     if (para_plain_A.dim() == 2):
#         rand_np = np.random.choice([0, 1], size=(para_plain_A.size(0) , para_plain_A.size(1)), p=[1 - RRAM_noise, RRAM_noise])
#     if (para_plain_A.dim() == 1):
#         rand_np = np.random.choice([0, 1], size=(para_plain_A.size(0)), p=[1 - RRAM_noise, RRAM_noise])
#     rand_tensor =torch.from_numpy(rand_np)
#     if (para_plain_A.dim() == 2):
#         rand_tensor = torch.round(torch.rand(para_plain_A.size(0) , para_plain_A.size(1)) * rand_tensor * quan_precision * 4)
#     if (para_plain_A.dim() == 1):
#         rand_tensor = torch.round(torch.rand(para_plain_A.shape[0]) * rand_tensor * quan_precision * 4)
#     # rand_tensor = torch.round(torch.rand(para_plain_A.shape[0], para_plain_A.shape[1]) * rand_tensor * quan_precision * 4)
#     sum_plain = (para_plain_A + para_plain_B + para_plain_C + para_plain_D + rand_tensor) % (quan_precision * 4)
#     return sum_plain

def SUM_w_enc(para_plain_A , para_plain_B , para_plain_C , para_plain_D):
    A = np.resize(para_plain_A, (8, 64))
    B = np.resize(para_plain_B, (8, 64))
    C = np.resize(para_plain_C, (8, 64))
    D = np.resize(para_plain_D, (8, 64))
    sum = np.zeros([8,64])
    for i in range (8):
        sum[i,:] =  FedAvg_PrivatePreserving(A[i,:], B[i,:],C[i,:], D[i,:], sk_c1, sk_c2, sk_c3, sk_c4, pk)
    sum_flat = sum.flatten()
    sum_flat = sum_flat[:482]

    return sum_flat

def SUM_w_enc_w_RRAM(para_plain_A , para_plain_B , para_plain_C , para_plain_D):
    A = np.resize(para_plain_A, (8, 64))
    B = np.resize(para_plain_B, (8, 64))
    C = np.resize(para_plain_C, (8, 64))
    D = np.resize(para_plain_D, (8, 64))
    sum = np.zeros([8,64])
    for i in range (8):
        sum[i,:] =  FedAvg_PrivatePreserving_with_RRAM(A[i,:], B[i,:],C[i,:], D[i,:], sk_c1, sk_c2, sk_c3, sk_c4, pk)
    sum_flat = sum.flatten()
    sum_flat = sum_flat[:482]
    return sum_flat

def FedAvg(model_AGG_FL, model_A_FL, model_B_FL, model_C_FL, model_D_FL, num_features):
    model_AGG_FL_FUNC = MyLSTM(num_features)
    para_dict = model_AGG_FL_FUNC.state_dict()
    for para_tensor in model_AGG_FL_FUNC.state_dict():
        para_value_A = model_A_FL.state_dict()[para_tensor]
        para_value_B = model_B_FL.state_dict()[para_tensor]
        para_value_C = model_C_FL.state_dict()[para_tensor]
        para_value_D = model_D_FL.state_dict()[para_tensor]
        para_value_AGG = (para_value_A + para_value_B + para_value_C + para_value_D) / 4
        para_dict[para_tensor] = para_value_AGG
    ## 不能在for循环中赋值，只能在循环外，因为load_state_dict()是深拷贝，而state_dict()是浅拷贝
    model_AGG_FL_FUNC.load_state_dict(para_dict)
    model_A_FL.load_state_dict(para_dict)
    model_B_FL.load_state_dict(para_dict)
    model_C_FL.load_state_dict(para_dict)
    model_D_FL.load_state_dict(para_dict)
    model_AGG_FL.load_state_dict(para_dict)
    return 

# def FedAvg_w_enc(model_AGG_FL_wquan, model_A_FL, model_B_FL, model_C_FL, model_D_FL, num_features, quan_precision):
#     model_AGG_FL_FUNC = MyLSTM(num_features)
#     para_dict = model_AGG_FL_FUNC.state_dict()
#     delta_factor = quan_precision * 15 ##  权重差的放大倍数

#     # 记录每个 tensor 的形状，以便稍后恢复它们的形状
#     shapes = {key: value.shape for key, value in model_A_FL.state_dict().items()}
#     para_value_A = np.concatenate([t.numpy().flatten() for t in model_A_FL.state_dict().values()])
#     para_value_B = np.concatenate([t.numpy().flatten() for t in model_B_FL.state_dict().values()])
#     para_value_C = np.concatenate([t.numpy().flatten() for t in model_C_FL.state_dict().values()])
#     para_value_D = np.concatenate([t.numpy().flatten() for t in model_D_FL.state_dict().values()])
#     para_value_AGG_FL_wquan = np.concatenate([t.numpy().flatten() for t in model_AGG_FL_wquan.state_dict().values()])
#     para_delta_A = para_value_A - para_value_AGG_FL_wquan
#     para_delta_B = para_value_B - para_value_AGG_FL_wquan
#     para_delta_C = para_value_C - para_value_AGG_FL_wquan
#     para_delta_D = para_value_D - para_value_AGG_FL_wquan

#     ## 将权重差放大 100 倍后再量化到 16 以内，超出 16 的按照 16 算,作为加密的明文
#     ## delta_factor/2 是 offset
#     sixteen = np.ones_like(para_delta_A) * quan_precision - 1
#     zero = np.zeros_like(para_delta_A)
#     para_plain_A = np.round(para_delta_A * delta_factor + quan_precision/2)
#     # max_plain = torch.max(para_plain_A)
#     # min_plain = torch.min(para_plain_A)
#     para_plain_A = np.where(para_plain_A > quan_precision - 1, sixteen, para_plain_A)
#     para_plain_A = np.where(para_plain_A < 0, zero, para_plain_A)

#     para_plain_B = np.round(para_delta_B * delta_factor + quan_precision/2)
#     # if (torch.max(para_plain_B) > max_plain):
#     #     max_plain = torch.max(para_plain_B)
#     # if (torch.min(para_plain_B) < min_plain):
#     #     min_plain = torch.min(para_plain_B)
#     para_plain_B = np.where(para_plain_B > quan_precision - 1, sixteen, para_plain_B)
#     para_plain_B = np.where(para_plain_B < 0, zero, para_plain_B)

#     para_plain_C = np.round(para_delta_C * delta_factor + quan_precision/2)
#     # if (torch.max(para_plain_C) > max_plain):
#     #     max_plain = torch.max(para_plain_C)
#     # if (torch.min(para_plain_C) < min_plain):
#     #     min_plain = torch.min(para_plain_C)
#     para_plain_C = np.where(para_plain_C > quan_precision - 1, sixteen, para_plain_C)
#     para_plain_C = np.where(para_plain_C < 0, zero, para_plain_C)

#     para_plain_D = np.round(para_delta_D * delta_factor + quan_precision/2)
#     # if (torch.max(para_plain_D) > max_plain):
#     #     max_plain = torch.max(para_plain_D)
#     # if (torch.min(para_plain_D) < min_plain):
#     #     min_plain = torch.min(para_plain_D)
#     para_plain_D = np.where(para_plain_D > quan_precision - 1, sixteen, para_plain_D)
#     para_plain_D = np.where(para_plain_D < 0, zero, para_plain_D)

#     # para_plain_SUM = para_plain_A + para_plain_B + para_plain_C + para_plain_D
#     para_plain_SUM = SUM_w_enc(para_plain_A, para_plain_B, para_plain_C, para_plain_D)
#     # para_plain_SUM_noerror = para_plain_A, para_plain_B, para_plain_C, para_plain_D
#     para_plain_SUM = para_value_AGG_FL_wquan + (para_plain_SUM - 4 * quan_precision/2)/ delta_factor / 4
#     ## 理想值
#     para_plain_true = (para_value_A + para_value_B + para_value_C + para_value_D) / 4
#     para_plain_error_w_true = para_plain_SUM - para_plain_true

#     # 将处理后的 Numpy 数组切片并放回字典中
#     start = 0
#     for key in para_dict.keys():
#         length = np.prod(shapes[key])  # 计算每个形状的元素总数
#         para_dict[key] = torch.from_numpy(para_plain_SUM[start:start+length].reshape(shapes[key]))
#         start += length

#     ## 不能在for循环中赋值，只能在循环外，因为load_state_dict()是深拷贝，而state_dict()是浅拷贝
#     model_AGG_FL_FUNC.load_state_dict(para_dict)
#     model_A_FL.load_state_dict(para_dict)
#     model_B_FL.load_state_dict(para_dict)
#     model_C_FL.load_state_dict(para_dict)
#     model_D_FL.load_state_dict(para_dict)
#     model_AGG_FL_wquan.load_state_dict(para_dict)
#     return para_plain_error_w_true

def FedAvg_w_enc_w_RRAM(model_AGG_FL_wquan, model_A_FL, model_B_FL, model_C_FL, model_D_FL, num_features, quan_precision):
    model_AGG_FL_FUNC = MyLSTM(num_features)
    para_dict = model_AGG_FL_FUNC.state_dict()
    delta_factor = quan_precision * 15 ##  权重差的放大倍数
    # delta_factor = quan_precision * 3 ##  权重差的放大倍数

    # 记录每个 tensor 的形状，以便稍后恢复它们的形状
    shapes = {key: value.shape for key, value in model_A_FL.state_dict().items()}
    para_value_A = np.concatenate([t.numpy().flatten() for t in model_A_FL.state_dict().values()])
    para_value_B = np.concatenate([t.numpy().flatten() for t in model_B_FL.state_dict().values()])
    para_value_C = np.concatenate([t.numpy().flatten() for t in model_C_FL.state_dict().values()])
    para_value_D = np.concatenate([t.numpy().flatten() for t in model_D_FL.state_dict().values()])
    para_value_AGG_FL_wquan = np.concatenate([t.numpy().flatten() for t in model_AGG_FL_wquan.state_dict().values()])
    para_delta_A = para_value_A - para_value_AGG_FL_wquan
    para_delta_B = para_value_B - para_value_AGG_FL_wquan
    para_delta_C = para_value_C - para_value_AGG_FL_wquan
    para_delta_D = para_value_D - para_value_AGG_FL_wquan

    ## 将权重差放大 100 倍后再量化到 16 以内，超出 16 的按照 16 算,作为加密的明文
    ## delta_factor/2 是 offset
    sixteen = np.ones_like(para_delta_A) * quan_precision - 1
    zero = np.zeros_like(para_delta_A)
    para_plain_A = np.round(para_delta_A * delta_factor + quan_precision/2)
    # max_plain = torch.max(para_plain_A)
    # min_plain = torch.min(para_plain_A)
    para_plain_A = np.where(para_plain_A > quan_precision - 1, sixteen, para_plain_A)
    para_plain_A = np.where(para_plain_A < 0, zero, para_plain_A)

    para_plain_B = np.round(para_delta_B * delta_factor + quan_precision/2)
    # if (torch.max(para_plain_B) > max_plain):
    #     max_plain = torch.max(para_plain_B)
    # if (torch.min(para_plain_B) < min_plain):
    #     min_plain = torch.min(para_plain_B)
    para_plain_B = np.where(para_plain_B > quan_precision - 1, sixteen, para_plain_B)
    para_plain_B = np.where(para_plain_B < 0, zero, para_plain_B)

    para_plain_C = np.round(para_delta_C * delta_factor + quan_precision/2)
    # if (torch.max(para_plain_C) > max_plain):
    #     max_plain = torch.max(para_plain_C)
    # if (torch.min(para_plain_C) < min_plain):
    #     min_plain = torch.min(para_plain_C)
    para_plain_C = np.where(para_plain_C > quan_precision - 1, sixteen, para_plain_C)
    para_plain_C = np.where(para_plain_C < 0, zero, para_plain_C)

    para_plain_D = np.round(para_delta_D * delta_factor + quan_precision/2)
    # if (torch.max(para_plain_D) > max_plain):
    #     max_plain = torch.max(para_plain_D)
    # if (torch.min(para_plain_D) < min_plain):
    #     min_plain = torch.min(para_plain_D)
    para_plain_D = np.where(para_plain_D > quan_precision - 1, sixteen, para_plain_D)
    para_plain_D = np.where(para_plain_D < 0, zero, para_plain_D)

    para_plain_SUM = SUM_w_enc_w_RRAM(para_plain_A, para_plain_B, para_plain_C, para_plain_D)
    para_plain_SUM_TXT = (para_plain_SUM - 4 * quan_precision/2)/ delta_factor / 4
    para_plain_SUM = para_value_AGG_FL_wquan + (para_plain_SUM - 4 * quan_precision/2)/ delta_factor / 4
    
    para_plain_SUM_diGFL = SUM_w_enc(para_plain_A, para_plain_B, para_plain_C, para_plain_D)
    para_plain_SUM_diGFL_TXT = (para_plain_SUM_diGFL - 4 * quan_precision/2)/ delta_factor / 4
    para_plain_SUM_diGFL = para_value_AGG_FL_wquan + (para_plain_SUM_diGFL - 4 * quan_precision/2)/ delta_factor / 4

    # ## 理想值
    # para_plain_SUM_quan = para_plain_A + para_plain_B + para_plain_C + para_plain_D
    # para_plain_SUM_quan = para_value_AGG_FL_wquan + (para_plain_SUM_quan - 4 * quan_precision/2)/ delta_factor / 4
    # para_plain_error_w_ideal = para_plain_SUM - para_plain_SUM_diGFL

    para_plain_true = (para_value_A + para_value_B + para_value_C + para_value_D) / 4
    para_plain_error_w_true = para_plain_SUM - para_plain_true
    para_plain_error_w_ideal = para_plain_SUM_diGFL - para_plain_true

    with open(" para_value_A.txt",'w') as f:    
        np.savetxt(f, para_value_A, fmt = '%f')
    with open(" para_value_B.txt",'w') as f:    
        np.savetxt(f, para_value_B, fmt = '%f')
    with open(" para_value_C.txt",'w') as f:    
        np.savetxt(f, para_value_C, fmt = '%f')
    with open(" para_value_D.txt",'w') as f:    
        np.savetxt(f, para_value_D, fmt = '%f')
    with open(" para_value_AGG_FL_wquan.txt",'w') as f:    
        np.savetxt(f, para_value_AGG_FL_wquan, fmt = '%f')
    with open(" para_plain_true.txt",'w') as f:    
        np.savetxt(f, para_plain_true, fmt = '%f')
    with open(" para_plain_SUM.txt",'w') as f:    
        np.savetxt(f, para_plain_SUM, fmt = '%f')
    with open(" para_plain_SUM_diGFL.txt",'w') as f:    
        np.savetxt(f, para_plain_SUM_diGFL, fmt = '%f')

    # with open(" para_plain_SUM_diGFL.txt",'w') as f:    
    #     np.savetxt(f, para_plain_SUM_diGFL, fmt = '%f')

    # # with open(" para_plain_SUM_quan.txt",'w') as f:    
    # #     np.savetxt(f, para_plain_SUM_quan, fmt = '%f')

    # with open(" para_plain_true.txt",'w') as f:    
    #     np.savetxt(f, para_plain_true, fmt = '%f')

    # with open(" para_plain_true.txt",'w') as f:    
    #     np.savetxt(f, para_plain_true, fmt = '%f')
    

    # 将处理后的 Numpy 数组切片并放回字典中
    start = 0
    for key in para_dict.keys():
        length = np.prod(shapes[key])  # 计算每个形状的元素总数
        para_dict[key] = torch.from_numpy(para_plain_SUM[start:start+length].reshape(shapes[key]))
        start += length

    ## 不能在for循环中赋值，只能在循环外，因为load_state_dict()是深拷贝，而state_dict()是浅拷贝
    model_AGG_FL_FUNC.load_state_dict(para_dict)
    model_A_FL.load_state_dict(para_dict)
    model_B_FL.load_state_dict(para_dict)
    model_C_FL.load_state_dict(para_dict)
    model_D_FL.load_state_dict(para_dict)
    model_AGG_FL_wquan.load_state_dict(para_dict)
    return para_value_AGG_FL_wquan, para_plain_SUM

def FedAvg_w_enc(model_AGG_FL_wquan, model_A_FL, model_B_FL, model_C_FL, model_D_FL, num_features, quan_precision):
    model_AGG_FL_FUNC = MyLSTM(num_features)
    para_dict = model_AGG_FL_FUNC.state_dict()
    delta_factor = quan_precision * 15 ##  权重差的放大倍数

    # 记录每个 tensor 的形状，以便稍后恢复它们的形状
    shapes = {key: value.shape for key, value in model_A_FL.state_dict().items()}
    para_value_A = np.concatenate([t.numpy().flatten() for t in model_A_FL.state_dict().values()])
    para_value_B = np.concatenate([t.numpy().flatten() for t in model_B_FL.state_dict().values()])
    para_value_C = np.concatenate([t.numpy().flatten() for t in model_C_FL.state_dict().values()])
    para_value_D = np.concatenate([t.numpy().flatten() for t in model_D_FL.state_dict().values()])
    para_value_AGG_FL_wquan = np.concatenate([t.numpy().flatten() for t in model_AGG_FL_wquan.state_dict().values()])
    para_delta_A = para_value_A - para_value_AGG_FL_wquan
    para_delta_B = para_value_B - para_value_AGG_FL_wquan
    para_delta_C = para_value_C - para_value_AGG_FL_wquan
    para_delta_D = para_value_D - para_value_AGG_FL_wquan

    ## 将权重差放大 100 倍后再量化到 16 以内，超出 16 的按照 16 算,作为加密的明文
    ## delta_factor/2 是 offset
    sixteen = np.ones_like(para_delta_A) * quan_precision - 1
    zero = np.zeros_like(para_delta_A)
    para_plain_A = np.round(para_delta_A * delta_factor + quan_precision/2)
    # max_plain = torch.max(para_plain_A)
    # min_plain = torch.min(para_plain_A)
    para_plain_A = np.where(para_plain_A > quan_precision - 1, sixteen, para_plain_A)
    para_plain_A = np.where(para_plain_A < 0, zero, para_plain_A)

    para_plain_B = np.round(para_delta_B * delta_factor + quan_precision/2)
    # if (torch.max(para_plain_B) > max_plain):
    #     max_plain = torch.max(para_plain_B)
    # if (torch.min(para_plain_B) < min_plain):
    #     min_plain = torch.min(para_plain_B)
    para_plain_B = np.where(para_plain_B > quan_precision - 1, sixteen, para_plain_B)
    para_plain_B = np.where(para_plain_B < 0, zero, para_plain_B)

    para_plain_C = np.round(para_delta_C * delta_factor + quan_precision/2)
    # if (torch.max(para_plain_C) > max_plain):
    #     max_plain = torch.max(para_plain_C)
    # if (torch.min(para_plain_C) < min_plain):
    #     min_plain = torch.min(para_plain_C)
    para_plain_C = np.where(para_plain_C > quan_precision - 1, sixteen, para_plain_C)
    para_plain_C = np.where(para_plain_C < 0, zero, para_plain_C)

    para_plain_D = np.round(para_delta_D * delta_factor + quan_precision/2)
    # if (torch.max(para_plain_D) > max_plain):
    #     max_plain = torch.max(para_plain_D)
    # if (torch.min(para_plain_D) < min_plain):
    #     min_plain = torch.min(para_plain_D)
    para_plain_D = np.where(para_plain_D > quan_precision - 1, sixteen, para_plain_D)
    para_plain_D = np.where(para_plain_D < 0, zero, para_plain_D)

    para_plain_SUM = SUM_w_enc(para_plain_A, para_plain_B, para_plain_C, para_plain_D)
    para_plain_SUM = para_value_AGG_FL_wquan + (para_plain_SUM - 4 * quan_precision/2)/ delta_factor / 4

    # ## 理想值
    para_plain_SUM_quan = para_plain_A + para_plain_B + para_plain_C + para_plain_D
    para_plain_SUM_quan = para_value_AGG_FL_wquan + (para_plain_SUM_quan - 4 * quan_precision/2)/ delta_factor / 4
    # para_plain_error_w_ideal = para_plain_SUM - para_plain_SUM_diGFL

    para_plain_true = (para_value_A + para_value_B + para_value_C + para_value_D) / 4
    para_plain_error_w_true = para_plain_SUM - para_plain_true
    para_plain_error_w_ideal = para_plain_SUM - para_plain_SUM_quan
    # print("max plain:" , max_plain/quan_precision)
    # print("min plain:" , min_plain/quan_precision)
    # print(para_plain_error)

    # with open(" para_plain_error_w_true.txt",'w') as f:    
    #     np.savetxt(f, para_plain_error_w_true, fmt = '%f')

    # with open(" para_plain_error_w_ideal.txt",'w') as f:    
    #     np.savetxt(f, para_plain_error_w_ideal, fmt = '%f')

    # with open(" para_plain_SUM_quan.txt",'w') as f:    
    #     np.savetxt(f, para_plain_SUM_quan, fmt = '%f')

    # with open(" para_plain_true.txt",'w') as f:    
    #     np.savetxt(f, para_plain_true, fmt = '%f')

    # with open(" para_plain_true.txt",'w') as f:    
    #     np.savetxt(f, para_plain_true, fmt = '%f')
    
    # 将处理后的 Numpy 数组切片并放回字典中
    start = 0
    for key in para_dict.keys():
        length = np.prod(shapes[key])  # 计算每个形状的元素总数
        para_dict[key] = torch.from_numpy(para_plain_SUM[start:start+length].reshape(shapes[key]))
        start += length

    ## 不能在for循环中赋值，只能在循环外，因为load_state_dict()是深拷贝，而state_dict()是浅拷贝
    model_AGG_FL_FUNC.load_state_dict(para_dict)
    model_A_FL.load_state_dict(para_dict)
    model_B_FL.load_state_dict(para_dict)
    model_C_FL.load_state_dict(para_dict)
    model_D_FL.load_state_dict(para_dict)
    model_AGG_FL_wquan.load_state_dict(para_dict)
    return para_plain_error_w_ideal, para_plain_error_w_true

def one_test(seed):
    quan_precision = 32
    torch.manual_seed(seed)
    NUM_EPOCHS = 11  #### 60
    CLIENT_DATA_NUM = 32 
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    learning_rate = 0.02 ## 0.02
    para_plain_error_w_ideal0 = np.zeros([NUM_EPOCHS,482])
    para_plain_error_w_true0 = np.zeros([NUM_EPOCHS,482])
    para_plain_error_w_ideal1 = np.zeros([NUM_EPOCHS,482])
    para_plain_error_w_true1 = np.zeros([NUM_EPOCHS,482])
    para_plain_error_w_ideal2 = np.zeros([NUM_EPOCHS,482])
    para_plain_error_w_true2 = np.zeros([NUM_EPOCHS,482])


    PATH_TRAIN_SEQS = "./data/sepsis/processed_data/sepsis.seqs.train"
    PATH_TRAIN_LABELS = "./data/sepsis/processed_data/sepsis.labels.train"
    PATH_TEST_SEQS = "./data/sepsis/processed_data/sepsis.seqs.test"
    PATH_TEST_LABELS = "./data/sepsis/processed_data/sepsis.labels.test"
    PATH_OUTPUT = "./out/"
    
    os.makedirs(PATH_OUTPUT, exist_ok=True)

    # Data loading
    print('===> Loading entire datasets')
    train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
    train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
    test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
    test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

    num_features = calculate_num_features(train_seqs)

    train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels)
    test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels)

    # 将数据集按照独立同分布分为4个参与方
    train_dataset_A=Subset(train_dataset,range(0,CLIENT_DATA_NUM))
    train_dataset_B=Subset(train_dataset,range(CLIENT_DATA_NUM, CLIENT_DATA_NUM*2))
    train_dataset_C=Subset(train_dataset,range(CLIENT_DATA_NUM*2, CLIENT_DATA_NUM*3))
    train_dataset_D=Subset(train_dataset,range(CLIENT_DATA_NUM*3, CLIENT_DATA_NUM*4))
    train_dataset_SUM=Subset(train_dataset,range(CLIENT_DATA_NUM, CLIENT_DATA_NUM*4))

    # 为每一方建立train_loader
    train_loader_A = DataLoader(dataset=train_dataset_A, batch_size=BATCH_SIZE, shuffle=False, collate_fn=time_collate_fn, num_workers=NUM_WORKERS)
    train_loader_B = DataLoader(dataset=train_dataset_B, batch_size=BATCH_SIZE, shuffle=False, collate_fn=time_collate_fn, num_workers=NUM_WORKERS)
    train_loader_C = DataLoader(dataset=train_dataset_C, batch_size=BATCH_SIZE, shuffle=False, collate_fn=time_collate_fn, num_workers=NUM_WORKERS)
    train_loader_D = DataLoader(dataset=train_dataset_D, batch_size=BATCH_SIZE, shuffle=False, collate_fn=time_collate_fn, num_workers=NUM_WORKERS)
    train_loader_SUM = DataLoader(dataset=train_dataset_SUM, batch_size=BATCH_SIZE, shuffle=False, collate_fn=time_collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=time_collate_fn,
                            num_workers=NUM_WORKERS)

    criterion = nn.CrossEntropyLoss()

    model_SUM = MyLSTM(num_features)
    optimizer_SUM = optim.Adam(model_SUM.parameters(), lr=learning_rate)

    model_AGG_FL = MyLSTM(num_features)

    model_A_FL = MyLSTM(num_features)
    optimizer_A_FL = optim.Adam(model_A_FL.parameters(), lr=learning_rate)
    model_B_FL = MyLSTM(num_features)
    optimizer_B_FL = optim.Adam(model_B_FL.parameters(), lr=learning_rate)
    model_C_FL = MyLSTM(num_features)
    optimizer_C_FL = optim.Adam(model_C_FL.parameters(), lr=learning_rate)
    model_D_FL = MyLSTM(num_features)
    optimizer_D_FL = optim.Adam(model_D_FL.parameters(), lr=learning_rate)

    model_AGG_FL_wquan = MyLSTM(num_features)
    model_AGG_FL_wRRAM = MyLSTM(num_features)
    model_AGG_FL_wRRAM_t2 = MyLSTM(num_features)

    ## service给各方分发模型
    para_dict = model_AGG_FL.state_dict()
    model_A_FL.load_state_dict(para_dict)
    model_B_FL.load_state_dict(para_dict)
    model_C_FL.load_state_dict(para_dict)
    model_D_FL.load_state_dict(para_dict)
    model_AGG_FL_wquan.load_state_dict(para_dict)
    model_AGG_FL_wRRAM.load_state_dict(para_dict)
    model_AGG_FL_wRRAM_t2.load_state_dict(para_dict)

    device = torch.device("cpu")
    model_SUM.to(device)
    model_AGG_FL.to(device)
    model_A_FL.to(device)
    model_B_FL.to(device)
    model_C_FL.to(device)
    model_D_FL.to(device)
    model_AGG_FL_wquan.to(device)
    model_AGG_FL_wRRAM.to(device)
    model_AGG_FL_wRRAM_t2.to(device)
    criterion.to(device)

    ## 5 Centralized training
    ## 6 digital FL
    ## 7 FL with precision loss
    ## 8 FL with precision loss and RRAM noise
    ## 9 FL with precision loss and RRAM noise * 2 
    train_loss = np.zeros((NUM_EPOCHS,9))
    train_accuracy = np.zeros((NUM_EPOCHS,9))
    test_accuracy = np.zeros((NUM_EPOCHS,9))
    ####### CL
    # for epoch in range(NUM_EPOCHS):
    #     train_loss[epoch,4], train_accuracy[epoch,4] = train(model_SUM, device, train_loader_SUM, criterion, optimizer_SUM, epoch)
    #     _, test_accuracy[epoch,4], _ = evaluate(model_SUM, device, test_loader, criterion)
    #     print("seed:",seed,"  CL  ","epoch:",epoch)
    # ####### digital FL
    # for epoch in range(NUM_EPOCHS):
    #     ## 各方自己训练
    #     train_loss_A, train_accuracy_A = train(model_A_FL, device, train_loader_A, criterion, optimizer_A_FL, epoch)
    #     train_loss_B, train_accuracy_B = train(model_B_FL, device, train_loader_B, criterion, optimizer_B_FL, epoch)
    #     train_loss_C, train_accuracy_C = train(model_C_FL, device, train_loader_C, criterion, optimizer_C_FL, epoch)
    #     train_loss_D, train_accuracy_D = train(model_D_FL, device, train_loader_D, criterion, optimizer_D_FL, epoch)
    #     train_loss[epoch,5] = (train_loss_A + train_loss_B + train_loss_C + train_loss_D)/4
    #     train_accuracy[epoch,5] = (train_accuracy_A + train_accuracy_B + train_accuracy_C + train_accuracy_D)/4
    #     ## service将各方训练得到的参数取平均，并下发给各方
    #     FedAvg(model_AGG_FL, model_A_FL, model_B_FL, model_C_FL, model_D_FL, num_features)
    #     _, test_accuracy[epoch,5], _ = evaluate(model_AGG_FL, device, test_loader, criterion)
    #     print("seed:",seed,"  dFL  ","epoch:",epoch)

    ####### 带加解密精度损失的 FL
    # para_dict = model_AGG_FL_wquan.state_dict()
    # model_A_FL.load_state_dict(para_dict)
    # model_B_FL.load_state_dict(para_dict)
    # model_C_FL.load_state_dict(para_dict)
    # model_D_FL.load_state_dict(para_dict)
    # for epoch in range(NUM_EPOCHS):
    #     ## 各方自己训练
    #     train_loss_A, train_accuracy_A = train(model_A_FL, device, train_loader_A, criterion, optimizer_A_FL, epoch)
    #     train_loss_B, train_accuracy_B = train(model_B_FL, device, train_loader_B, criterion, optimizer_B_FL, epoch)
    #     train_loss_C, train_accuracy_C = train(model_C_FL, device, train_loader_C, criterion, optimizer_C_FL, epoch)
    #     train_loss_D, train_accuracy_D = train(model_D_FL, device, train_loader_D, criterion, optimizer_D_FL, epoch)
    #     train_loss[epoch,6] = (train_loss_A + train_loss_B + train_loss_C + train_loss_D)/4
    #     train_accuracy[epoch,6] = (train_accuracy_A + train_accuracy_B + train_accuracy_C + train_accuracy_D)/4  
    #     ## service将各方训练得到的参数取平均，并下发给各方
    #     para_plain_error_w_ideal0[epoch][:],para_plain_error_w_true0[epoch][:] = FedAvg_w_enc(model_AGG_FL_wquan, model_A_FL, model_B_FL, model_C_FL, model_D_FL, num_features, quan_precision)
    #     _, test_accuracy[epoch,6], _ = evaluate(model_AGG_FL_wquan, device, test_loader, criterion)
    #     print("seed:",seed,"  encFL  ","epoch:",epoch)

    ####### 带精度损失 + RRAM引起随机崩坏的 FL
    para_dict = model_AGG_FL_wRRAM.state_dict()
    model_A_FL.load_state_dict(para_dict)
    model_B_FL.load_state_dict(para_dict)
    model_C_FL.load_state_dict(para_dict)
    model_D_FL.load_state_dict(para_dict)
    for epoch in range(NUM_EPOCHS):
        ## 各方自己训练
        # for i in range()
        train_loss_A, train_accuracy_A = train(model_A_FL, device, train_loader_A, criterion, optimizer_A_FL, epoch)
        train_loss_B, train_accuracy_B = train(model_B_FL, device, train_loader_B, criterion, optimizer_B_FL, epoch)
        train_loss_C, train_accuracy_C = train(model_C_FL, device, train_loader_C, criterion, optimizer_C_FL, epoch)
        train_loss_D, train_accuracy_D = train(model_D_FL, device, train_loader_D, criterion, optimizer_D_FL, epoch)
        train_loss[epoch,7] = (train_loss_A + train_loss_B + train_loss_C + train_loss_D)/4
        train_accuracy[epoch,7] = (train_accuracy_A + train_accuracy_B + train_accuracy_C + train_accuracy_D)/4
        ## service将各方训练得到的参数取平均，并下发给各方
        # FedAvg_wRRAM(model_AGG_FL_wRRAM, model_A_FL, model_B_FL, model_C_FL, model_D_FL, num_features, quan_precision, RRAM_noise)
        if(epoch % 10 == 0):
            para_plain_error_w_ideal1[epoch][:], para_plain_error_w_true1[epoch][:] = FedAvg_w_enc_w_RRAM(model_AGG_FL_wRRAM, model_A_FL, model_B_FL, model_C_FL, model_D_FL, num_features, quan_precision)
        _, test_accuracy[epoch,7], _ = evaluate(model_AGG_FL_wRRAM, device, test_loader, criterion)
        print("seed:",seed,"  RRAMFL1  ","epoch:",epoch)

    # # ####### 带精度损失 + RRAM引起随机崩坏的 FL， 噪声 0.1
    # para_dict = model_AGG_FL_wRRAM_t2.state_dict()
    # model_A_FL.load_state_dict(para_dict)
    # model_B_FL.load_state_dict(para_dict)
    # model_C_FL.load_state_dict(para_dict)
    # model_D_FL.load_state_dict(para_dict)
    # for epoch in range(NUM_EPOCHS):
    #     ## 各方自己训练
    #     train_loss_A, train_accuracy_A = train(model_A_FL, device, train_loader_A, criterion, optimizer_A_FL, epoch)
    #     train_loss_B, train_accuracy_B = train(model_B_FL, device, train_loader_B, criterion, optimizer_B_FL, epoch)
    #     train_loss_C, train_accuracy_C = train(model_C_FL, device, train_loader_C, criterion, optimizer_C_FL, epoch)
    #     train_loss_D, train_accuracy_D = train(model_D_FL, device, train_loader_D, criterion, optimizer_D_FL, epoch)
    #     train_loss[epoch,8] = (train_loss_A + train_loss_B + train_loss_C + train_loss_D)/4
    #     train_accuracy[epoch,8] = (train_accuracy_A + train_accuracy_B + train_accuracy_C + train_accuracy_D)/4
    #     ## service将各方训练得到的参数取平均，并下发给各方
    #     para_plain_error_w_ideal2[epoch][:], para_plain_error_w_true2[epoch][:] = FedAvg_w_enc_w_RRAM(model_AGG_FL_wRRAM_t2, model_A_FL, model_B_FL, model_C_FL, model_D_FL, num_features, quan_precision)
    #     _, test_accuracy[epoch,8], _ = evaluate(model_AGG_FL_wRRAM_t2, device, test_loader, criterion)
    #     print("seed:",seed,"  RRAMFL2  ","epoch:",epoch)

    acc = np.zeros((1,5))
    pre = np.zeros((1,5))
    rec = np.zeros((1,5))
    f1s = np.zeros((1,5))
    roc = np.zeros((1,5))
    mcc = np.zeros((1,5))

    # print("\nEvaluation metrics on test set for client A B C D: \t")
    # acc[0,0],pre[0,0],rec[0,0],f1s[0,0],roc[0,0],mcc[0,0] = best_evaluate(model_SUM, device, test_loader)
    # acc[0,1],pre[0,1],rec[0,1],f1s[0,1],roc[0,1],mcc[0,1] = best_evaluate(model_AGG_FL, device, test_loader)
    # acc[0,2],pre[0,2],rec[0,2],f1s[0,2],roc[0,2],mcc[0,2] = best_evaluate(model_AGG_FL_wquan, device, test_loader)
    # acc[0,3],pre[0,3],rec[0,3],f1s[0,3],roc[0,3],mcc[0,3] = best_evaluate(model_AGG_FL_wRRAM, device, test_loader)
    # acc[0,4],pre[0,4],rec[0,4],f1s[0,4],roc[0,4],mcc[0,4] = best_evaluate(model_AGG_FL_wRRAM_t2, device, test_loader)

    # with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_ideal0.txt"),'ab') as f:
    #     np.savetxt(f, para_plain_error_w_ideal0.T, fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_true0.txt"),'ab') as f:
    #     np.savetxt(f, para_plain_error_w_true0.T, fmt = '%f')
    with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_ideal1.txt"),'ab') as f:
        np.savetxt(f, para_plain_error_w_ideal1.T, fmt = '%f')
    with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_true1.txt"),'ab') as f:
        np.savetxt(f, para_plain_error_w_true1.T, fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_ideal2.txt"),'ab') as f:
    #     np.savetxt(f, para_plain_error_w_ideal2.T, fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_true2.txt"),'ab') as f:
    #     np.savetxt(f, para_plain_error_w_true2.T, fmt = '%f')

    # with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_ideal0_1dim.txt"),'ab') as f:
    #     np.savetxt(f, para_plain_error_w_ideal0.flatten(), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_true0_1dim.txt"),'ab') as f:
    #     np.savetxt(f, para_plain_error_w_true0.flatten(), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_ideal1_1dim.txt"),'ab') as f:
    #     np.savetxt(f, para_plain_error_w_ideal1.flatten(), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_true1_1dim.txt"),'ab') as f:
    #     np.savetxt(f, para_plain_error_w_true1.flatten(), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_ideal2_1dim.txt"),'ab') as f:
    #     np.savetxt(f, para_plain_error_w_ideal2.flatten(), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " para_plain_error_w_true2_1dim.txt"),'ab') as f:
    #     np.savetxt(f, para_plain_error_w_true2.flatten(), fmt = '%f')

    # with open(os.path.join(PATH_OUTPUT, " train_loss for SUM.txt"),'ab') as f:
    #     np.savetxt(f, train_loss[:,4].reshape(1,-1), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " train_accuracy for SUM.txt"),'ab') as f:
    #     np.savetxt(f, train_accuracy[:,4].reshape(1,-1), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " test_accuracy for SUM.txt"),'ab') as f:    
    #     np.savetxt(f, test_accuracy[:,4].reshape(1,-1), fmt = '%f')

    # with open(os.path.join(PATH_OUTPUT, " train_loss for FL.txt"),'ab') as f:
    #     np.savetxt(f, train_loss[:,5].reshape(1,-1), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " train_accuracy for FL.txt"),'ab') as f:
    #     np.savetxt(f, train_accuracy[:,5].reshape(1,-1), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " test_accuracy for FL.txt"),'ab') as f:    
    #     np.savetxt(f, test_accuracy[:,5].reshape(1,-1), fmt = '%f')

    # with open(os.path.join(PATH_OUTPUT, " train_loss for FL with quantitation.txt"),'ab') as f:
    #     np.savetxt(f, train_loss[:,6].reshape(1,-1), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " train_accuracy for FL with quantitation.txt"),'ab') as f:
    #     np.savetxt(f, train_accuracy[:,6].reshape(1,-1), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " test_accuracy for FL with quantitation.txt"),'ab') as f:    
    #     np.savetxt(f, test_accuracy[:,6].reshape(1,-1), fmt = '%f')

    # with open(os.path.join(PATH_OUTPUT, " train_loss for FL with RRAM.txt"),'ab') as f:
    #     np.savetxt(f, train_loss[:,7].reshape(1,-1), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " train_accuracy for FL with RRAM .txt"),'ab') as f:
    #     np.savetxt(f, train_accuracy[:,7].reshape(1,-1), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " test_accuracy for FL with RRAM.txt"),'ab') as f:    
    #     np.savetxt(f, test_accuracy[:,7].reshape(1,-1), fmt = '%f')

    # with open(os.path.join(PATH_OUTPUT, " train_loss for FL with RRAM noise t2.txt"),'ab') as f:
    #     np.savetxt(f, train_loss[:,8].reshape(1,-1), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " train_accuracy for FL with RRAM noise t2.txt"),'ab') as f:
    #     np.savetxt(f, train_accuracy[:,8].reshape(1,-1), fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " test_accuracy for FL with RRAM noise t2.txt"),'ab') as f:    
    #     np.savetxt(f, test_accuracy[:,8].reshape(1,-1), fmt = '%f')

    # with open(os.path.join(PATH_OUTPUT, " ACC.txt"),'ab') as f:    
    #     np.savetxt(f, acc, fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " PRE.txt"),'ab') as f:    
    #     np.savetxt(f, pre, fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " REC.txt"),'ab') as f:    
    #     np.savetxt(f, rec, fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " F1S.txt"),'ab') as f:    
    #     np.savetxt(f, f1s, fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " ROC.txt"),'ab') as f:    
    #     np.savetxt(f, roc, fmt = '%f')
    # with open(os.path.join(PATH_OUTPUT, " MCC.txt"),'ab') as f:    
    #     np.savetxt(f, mcc, fmt = '%f')


for i in range (1):
    one_test(i)
    print(i)
