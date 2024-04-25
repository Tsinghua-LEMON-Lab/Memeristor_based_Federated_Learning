import numpy as np
import os
from enclib_CORE import *
from enclib_BASIC import *
from RNSlib import *
# import statsmodels.api as sm
# pt refers to plain text; ct refers to cipher text; both are encrypted data

def ctaddct(ct0, ct1): 
    res = homoadd(ct0, ct1)
    return res    

def ptmulct(pt, ct):   
    c0 = modmul(pt, ct[0])
    c1 = modmul(pt, ct[1])
    res = [[0]*N for _ in range(2)]
    for i in range(N):
        res[0][i] = c0[i]
        res[1][i] = c1[i]
    return res    

def FedAvg_PrivatePreserving(dw_c1, dw_c2, dw_c3, dw_c4, sk_c1, sk_c2, sk_c3, sk_c4, pk):
    # --- Each Client encrypt delta weights --- 
    dw_ct_c1 = encrypt(dw_c1, pk)
    dw_ct_c2 = encrypt(dw_c2, pk)
    dw_ct_c3 = encrypt(dw_c3, pk)
    dw_ct_c4 = encrypt(dw_c4, pk)

    # --- Server do addition ---
    dw_ct_sum = homoadd(dw_ct_c1, homoadd(dw_ct_c2, homoadd(dw_ct_c3, dw_ct_c4)))

    # --- Each clients do: Part decrytion ---
    dw_dct_c1 = part_decrypt(dw_ct_sum, sk_c1)
    dw_dct_c2 = part_decrypt(dw_ct_sum, sk_c2)
    dw_dct_c3 = part_decrypt(dw_ct_sum, sk_c3)
    dw_dct_c4 = part_decrypt(dw_ct_sum, sk_c4)

    # --- Server do: decryption merge ---
    dw_pt_sum = decrypt_merge(dw_dct_c1, dw_dct_c2, dw_dct_c3, dw_dct_c4)
    sum_ideal = dw_c1 + dw_c2 + dw_c3 + dw_c4
    for i in range (64):
        if(sum_ideal[i] != dw_pt_sum[i]):
            print("no")
    return dw_pt_sum

def FedAvg_PrivatePreserving_with_RRAM(dw_c1, dw_c2, dw_c3, dw_c4, sk_c1, sk_c2, sk_c3, sk_c4, pk):
    # --- Each Client encrypt delta weights --- 
    dw_ct_c1 = encrypt_wRRAM(dw_c1, pk)
    dw_ct_c2 = encrypt_wRRAM(dw_c2, pk)
    dw_ct_c3 = encrypt_wRRAM(dw_c3, pk)
    dw_ct_c4 = encrypt_wRRAM(dw_c4, pk)

    # --- Server do addition ---
    dw_ct_sum = homoadd(dw_ct_c1, homoadd(dw_ct_c2, homoadd(dw_ct_c3, dw_ct_c4)))

    # --- Each clients do: Part decrytion ---
    dw_dct_c1 = part_decrypt_wRRAM(dw_ct_sum, sk_c1)
    dw_dct_c2 = part_decrypt_wRRAM(dw_ct_sum, sk_c2)
    dw_dct_c3 = part_decrypt_wRRAM(dw_ct_sum, sk_c3)
    dw_dct_c4 = part_decrypt_wRRAM(dw_ct_sum, sk_c4)

    # --- Server do: decryption merge ---
    dw_pt_sum = decrypt_merge(dw_dct_c1, dw_dct_c2, dw_dct_c3, dw_dct_c4)
    return dw_pt_sum


def KeyGen():
    np.random.seed(0)
    sk_c1 = secret_keygen()
    sk_c2 = secret_keygen()
    sk_c3 = secret_keygen()
    sk_c4 = secret_keygen()

    # --- Server do: Public Key Seed generation ---
    pk_seed = public_key_seedgen()    

    # --- Each cliens do: Partial Public Key generation ---
    pk_c1 = part_public_keygen(sk_c1, pk_seed)
    pk_c2 = part_public_keygen(sk_c2, pk_seed)
    pk_c3 = part_public_keygen(sk_c3, pk_seed)
    pk_c4 = part_public_keygen(sk_c4, pk_seed)

    # --- Server do: Public Key merge ---
    pk = public_keymerge(pk_c1, pk_c2, pk_c3, pk_c4)

    sk_c1_mat = poly_to_mat_sk(sk_c1)
    sk_c2_mat = poly_to_mat_sk(sk_c2)
    sk_c3_mat = poly_to_mat_sk(sk_c3)
    sk_c4_mat = poly_to_mat_sk(sk_c4)

    pk0_mat = poly_to_mat_pk(pk[0])
    pk0_mat_T = [ [0] * N for _ in range(N) ]
    for i in range(N):
        for j in range (N):
            pk0_mat_T[i][j] = pk0_mat[j][i]
    pk0_mat_RNS = RNS_Decomposition_for_mat(pk0_mat_T)
    pk0_mat_RNS_bit = binexp_RNS_for_mat(pk0_mat_RNS)

    np_sk_c1 = np.array(sk_c1_mat).T
    np_sk_c2 = np.array(sk_c2_mat).T
    np_sk_c3 = np.array(sk_c3_mat).T
    np_sk_c4 = np.array(sk_c4_mat).T
    np_pk0 = np.array(pk0_mat_RNS_bit)

    # print(np_sk_c1.shape)
    with open('./out/key storage/sk_c1_mat.txt','w') as f:
        np.savetxt(f, np_sk_c1, fmt = '%d')
    with open('./out/key storage/sk_c2_mat.txt','w') as f:
        np.savetxt(f, np_sk_c2, fmt = '%d')
    with open('./out/key storage/sk_c3_mat.txt','w') as f:
        np.savetxt(f, np_sk_c3, fmt = '%d')
    with open('./out/key storage/sk_c4_mat.txt','w') as f:
        np.savetxt(f, np_sk_c4, fmt = '%d')
    for i in range(bit_len):
        with open('./out/key storage/pk0_mat_RNS_bit'+'%d.txt'%i,'w') as f:
            np.savetxt(f, np_pk0[i,:,:], fmt = '%d') 

    return sk_c1, sk_c2, sk_c3, sk_c4, pk

def u_Gen():  ## 变成列表好调用
    np.random.seed(1)
    # u_list = np.random.randint(0, 2, size = [64,1000], dtype = np.uint64)  
    u_list = np.random.randint(0, 2, size = [64,1000], dtype = np.uint64) 
    with open('./out/key storage/u_list.txt','w') as f:
        np.savetxt(f, u_list, fmt = '%d')
    return u_list.tolist()

def ct_Gen(i_seed):  ## 变成列表好调用
    np.random.seed(i_seed)
    ct_list = np.random.randint(0, Q, size = [64,1000], dtype = np.uint64) 
    with open('./out/key storage/ct_list_'+'%d.txt'%i_seed,'w') as f:
        np.savetxt(f, ct_list, fmt = '%d')
    return ct_list.tolist()

# --- Tests ---
def tests():
    # --- Each cliens do: Secret Key generation ---
    sk_c1 = secret_keygen()
    sk_c2 = secret_keygen()
    sk_c3 = secret_keygen()
    sk_c4 = secret_keygen()

    # --- Server do: Public Key Seed generation ---
    pk_seed = public_key_seedgen()    

    # --- Each cliens do: Partial Public Key generation ---
    pk_c1 = part_public_keygen(sk_c1, pk_seed)
    pk_c2 = part_public_keygen(sk_c2, pk_seed)
    pk_c3 = part_public_keygen(sk_c3, pk_seed)
    pk_c4 = part_public_keygen(sk_c4, pk_seed)

    # --- Server do: Public Key merge ---
    pk = public_keymerge(pk_c1, pk_c2, pk_c3, pk_c4)

    # --- Each Client encrypt delta weights --- 
    dw_c1 = [3] * 3 + [0] * (N-3)
    dw_c2 = [3] * 3 + [0] * (N-3)
    dw_c3 = [3] * 3 + [0] * (N-3)
    dw_c4 = [3] * 3 + [0] * (N-3)
    dw_ct_c1 = encrypt(dw_c1, pk)
    dw_ct_c2 = encrypt(dw_c2, pk)
    dw_ct_c3 = encrypt(dw_c3, pk)
    dw_ct_c4 = encrypt(dw_c4, pk)

    # --- Server do addition ---
    dw_ct_sum = homoadd(dw_ct_c1, homoadd(dw_ct_c2, homoadd(dw_ct_c3, dw_ct_c4)))

    # --- Each clients do: Part decrytion ---
    dw_dct_c1 = part_decrypt(dw_ct_sum, sk_c1)
    dw_dct_c2 = part_decrypt(dw_ct_sum, sk_c2)
    dw_dct_c3 = part_decrypt(dw_ct_sum, sk_c3)
    dw_dct_c4 = part_decrypt(dw_ct_sum, sk_c4)

    # --- Server do: decryption merge ---
    dw_pt_sum = decrypt_merge(dw_dct_c1, dw_dct_c2, dw_dct_c3, dw_dct_c4)
    print(dw_pt_sum)
    # assert dw_pt_sum[0] == (dw_c1[0] * dw_c2[0] + dw_c3[0] + dw_c4[0]) % t

def tests_ct_ACF():
    # --- Each cliens do: Secret Key generation ---
    sk_c1 = secret_keygen()
    sk_c2 = secret_keygen()
    sk_c3 = secret_keygen()
    sk_c4 = secret_keygen()

    # --- Server do: Public Key Seed generation ---
    pk_seed = public_key_seedgen()    

    # --- Each cliens do: Partial Public Key generation ---
    pk_c1 = part_public_keygen(sk_c1, pk_seed)
    pk_c2 = part_public_keygen(sk_c2, pk_seed)
    pk_c3 = part_public_keygen(sk_c3, pk_seed)
    pk_c4 = part_public_keygen(sk_c4, pk_seed)

    # --- Server do: Public Key merge ---
    pk = public_keymerge(pk_c1, pk_c2, pk_c3, pk_c4)

    # --- Each Client encrypt delta weights --- 
    dw_c1 = [3] * 3 + [0] * (N-3)
    
    NUM = 100
    out = np.zeros([NUM,N])
    for i in range (NUM):
        dw_ct_c1 = encrypt(dw_c1, pk)
        out[i][:] = dw_ct_c1[0]

    # out_corr_matrix = np.corrcoef(out.T, rowvar=False)
    # with open('./out/out_corr.txt','w') as f:
    #     np.savetxt(f, out_corr_matrix, fmt = '%f')

    with open('./out/CT_ACF.txt','w') as f:
        np.savetxt(f, out, fmt = '%d')
    # acf = sm.tsa.stattools.acf(out[:][0], nlags=NUM)
    # with open('./out/ACF.txt','w') as f:
    #     np.savetxt(f, acf, fmt = '%f')

    # a = np.random.randint(0, Q, size = [NUM * 2,N], dtype = np.uint64)
    # ran_acf = sm.tsa.stattools.acf(a.flatten(), nlags=NUM*100)
    # with open('./out/random_ACF.txt','w') as f:
    #     np.savetxt(f, ran_acf, fmt = '%f')


    


def tests_FedAvg():
    sk_c1, sk_c2, sk_c3, sk_c4, pk = KeyGen()
    u_Gen()

    dw_c1 = np.random.randint(0, t, size = N, dtype = np.uint64).tolist()
    dw_c2 = np.random.randint(0, t, size = N, dtype = np.uint64).tolist()
    dw_c3 = np.random.randint(0, t, size = N, dtype = np.uint64).tolist()
    dw_c4 = np.random.randint(0, t, size = N, dtype = np.uint64).tolist()
    dw_pt_sum_enc = FedAvg_PrivatePreserving(dw_c1, dw_c2, dw_c3, dw_c4, sk_c1, sk_c2, sk_c3, sk_c4, pk)
    dw_pt_sum = [0] * N
    for i in range(N):
            dw_pt_sum[i] = (dw_c1[i] + dw_c2[i] + dw_c3[i] + dw_c4[i]) % t
    # print(dw_pt_sum)
    # print(dw_pt_sum_enc)
    dw_pt_sum_enc_RRAM = FedAvg_PrivatePreserving_with_RRAM(dw_c1, dw_c2, dw_c3, dw_c4, sk_c1, sk_c2, sk_c3, sk_c4, pk)
    ERROR1 = [0] * N
    for i in range(N):
        ERROR1[i] = dw_pt_sum[i] - dw_pt_sum_enc[i]
    ERROR2 = [0] * N
    for i in range(N):
        ERROR2[i] = dw_pt_sum[i] - dw_pt_sum_enc_RRAM[i]
    print(ERROR1)
    print(ERROR2)

def for_tester_debug_pk():
    sk_c1, sk_c2, sk_c3, sk_c4, pk = KeyGen()
    pk0_mat = poly_to_mat_pk(pk[0])
    ## 转置
    pk0_mat_T = [ [0] * N for _ in range(N) ]
    for i in range(N):
        for j in range (N):
            pk0_mat_T[i][j] = pk0_mat[j][i]
    pk0_mat_RNS = RNS_Decomposition_for_mat(pk0_mat_T)
    pk0_mat_RNS_bit = binexp_RNS_for_mat(pk0_mat_RNS)
    u_list = u_Gen()
    u_num = 1000 ## 1000
    ### 结果应该为 8 * 1000 个，
    bitwidth = [0] * nRNSbase
    bitwidth_all = 0
    for i in range (nRNSbase):
        bitwidth[i] = math.ceil(math.log(RNSbase[i], 2))
        bitwidth_all = bitwidth_all + bitwidth[i]
    res_bit_RNS = [[ [0] * bitwidth_all for _ in range(u_num) ] for _ in range(8)]
    for q in range(8): ## 8 个 8 行
        for j in range(u_num):
            for i in range (bitwidth_all):
                for p in range (8): ## 8 行
                    res_bit_RNS[q][j][i] = res_bit_RNS[q][j][i] + pk0_mat_RNS_bit[i][p+q*8][0] * u_list[p+q*8][j] ## 有个转置，所以是 mat行 * vec列
    # bit to RNS
    res_RNS = [[ [0] * nRNSbase for _ in range(u_num) ] for _ in range(8)]
    for i in range(8):
        for j in range(u_num):
            res_RNS[i][j][0] = (res_bit_RNS[i][j][1] + res_bit_RNS[i][j][0] * 2) % 3;  #3
            res_RNS[i][j][1] = (res_bit_RNS[i][j][4] + res_bit_RNS[i][j][3] * 2 + res_bit_RNS[i][j][2] * 4) % 5  #5
            res_RNS[i][j][2] = (res_bit_RNS[i][j][7] + res_bit_RNS[i][j][6] * 2 + res_bit_RNS[i][j][5] * 4) % 7  #7
            res_RNS[i][j][3] = (res_bit_RNS[i][j][11] + res_bit_RNS[i][j][10] * 2 + res_bit_RNS[i][j][9] * 4 + res_bit_RNS[i][j][8] * 8) % 11  #11
            res_RNS[i][j][4] = (res_bit_RNS[i][j][15] + res_bit_RNS[i][j][14] * 2 + res_bit_RNS[i][j][13] * 4 + res_bit_RNS[i][j][12] * 8) % 13 #13
            res_RNS[i][j][5] = (res_bit_RNS[i][j][20] + res_bit_RNS[i][j][19] * 2 + res_bit_RNS[i][j][18] * 4 + res_bit_RNS[i][j][17] * 8 + res_bit_RNS[i][j][16] * 16) % 19  #19
            res_RNS[i][j][6] = (res_bit_RNS[i][j][25] + res_bit_RNS[i][j][24] * 2 + res_bit_RNS[i][j][23] * 4 + res_bit_RNS[i][j][22] * 8 + res_bit_RNS[i][j][21] * 16) % 23  #23
            res_RNS[i][j][7] = (res_bit_RNS[i][j][30] + res_bit_RNS[i][j][29] * 2 + res_bit_RNS[i][j][28] * 4 + res_bit_RNS[i][j][27] * 8 + res_bit_RNS[i][j][26] * 16) % 29  #29
            res_RNS[i][j][8] = (res_bit_RNS[i][j][35] + res_bit_RNS[i][j][34] * 2 + res_bit_RNS[i][j][33] * 4 + res_bit_RNS[i][j][32] * 8 + res_bit_RNS[i][j][31] * 16) % 32  #32
    # RNS to number
    res = [[0] * u_num for _ in range(8)]
    a = [0] * nRNScal
    for q in range (8):
        for i in range (u_num):
            for j in range (nRNScal):
                a[j] = res_RNS[q][i][j+RNScalbeg]
            res[q][i] = (RNS_Combination_for_number(a,RNScal,nRNScal)) % Q
    ## 自检错 因为代码运行有问题，但这里计算机不该有错
    b = [0] * nRNSred
    RNS_corr_flag = 0
    for q in range (8):
        for i in range (u_num):
            RNS_corr_flag = 0  
            for j in range (nRNSred):
                b[j] = res_RNS[q][i][j]
            for j in range (8):
                if (((res[q][i] + Q * j) % 3 == b[0]) and ((res[q][i] + Q * j) % 5 == b[1]) and ((res[q][i] + Q * j) % 7 == b[2])):
                    RNS_corr_flag = 1
            if (RNS_corr_flag == 0):
                print("RNS error")

    ## 1 个 8 行
    res_np = np.array(res).T
    for i in range (1,8):
        res_np[:,i] = (res_np[:,i] + res_np[:,i-1]) %Q
    with open('./out/res_np.txt','w') as f:
        np.savetxt(f, res_np, fmt = '%d')
    ## 8 个 8 行的和
    res_np_sum = (np.sum(res_np,1))% Q
    with open('./out/res_np_sum.txt','w') as f:
        np.savetxt(f, res_np_sum, fmt = '%d')

    ## 正确输出（校验过）
    res_correct = np.zeros([u_num,N])
    for i in range (u_num):
        # a = [x[i] for x in u_list]
        # b = np.array(tests_pk(pk[0], [x[i] for x in u_list]))
        res_correct[i,:] = np.array(tests_pk(pk[0], [x[i] for x in u_list])).T
    # res_correct_np = np.array(res_correct)
    with open('./out/res_correct_np.txt','w') as f:
        np.savetxt(f, res_correct, fmt = '%d')
    return

def for_tester_debug_sk():
    sk_c1, sk_c2, sk_c3, sk_c4, pk = KeyGen()
    sk_c1_mat = poly_to_mat_sk(sk_c1)
    ## 转置
    sk_c1_mat_T = [ [0] * N for _ in range(N) ]
    for i in range(N):
        for j in range (N):
            sk_c1_mat_T[i][j] = sk_c1_mat[j][i]

    ct_list_5 = ct_Gen(5)
    ct_list_89 = ct_Gen(89)
    ct_list_134 = ct_Gen(134)
    ct_list_6524 = ct_Gen(6524)

    ct_num = 1000 ## 1000

    ct_RNS_bit = [[ [0] * N for _ in range(36) ] for _ in range(ct_num)]
    a = [0] * 64
    for i in range (ct_num):
        for p in range (N):
            a[p] = ct_list_5[p][i]
        ct_RNS = RNS_Decomposition_for_vec(a) ## 只取第一个输入
        ct_RNS_bit_ori = binexp_RNS_for_vec(ct_RNS)  ## shape [36][64]
        for p in range (36):
            for q in range (64):
                ct_RNS_bit[i][p][q] = ct_RNS_bit_ori[p][q] ## shape [1000][36][64] 正确~
        
    ### 结果应该为 8 * 1000 个，
    bitwidth = [0] * nRNSbase
    bitwidth_all = 0
    for i in range (nRNSbase):
        bitwidth[i] = math.ceil(math.log(RNSbase[i], 2))
        bitwidth_all = bitwidth_all + bitwidth[i]
    res_bit_RNS = [[ [0] * bitwidth_all for _ in range(ct_num) ] for _ in range(8)]
    for q in range(8): ## 8 个 8 行
        for j in range(ct_num):
            for i in range (bitwidth_all):
                for p in range (8): ## 8 行
                    res_bit_RNS[q][j][i] = res_bit_RNS[q][j][i] + ct_RNS_bit[j][i][p+q*8] * sk_c1_mat_T[p+q*8][1] ## 有个转置，所以是 mat行 * vec列
    
    # for q in range(1): ## 8 个 8 行
    #     for j in range(ct_num):
    #         for i in range (bitwidth_all):
    #             for p in range (1,8): ## 8 行
    #                 res_bit_RNS[q][j][i] = res_bit_RNS[q][j][i] + ct_RNS_bit[j][i][p+q*8] * sk_c1_mat_T[p+q*8][0] ## 有个转置，所以是 mat行 * vec列

    # bit to RNS
    res_RNS = [[ [0] * nRNSbase for _ in range(ct_num) ] for _ in range(8)]
    for i in range(8):
        for j in range(ct_num):
            # a = res_bit_RNS[i][j][1] + res_bit_RNS[i][j][0] * 2
            res_RNS[i][j][0] = (res_bit_RNS[i][j][1] + res_bit_RNS[i][j][0] * 2) % 3  #3
            res_RNS[i][j][1] = (res_bit_RNS[i][j][4] + res_bit_RNS[i][j][3] * 2 + res_bit_RNS[i][j][2] * 4) % 5  #5
            res_RNS[i][j][2] = (res_bit_RNS[i][j][7] + res_bit_RNS[i][j][6] * 2 + res_bit_RNS[i][j][5] * 4) % 7  #7
            res_RNS[i][j][3] = (res_bit_RNS[i][j][11] + res_bit_RNS[i][j][10] * 2 + res_bit_RNS[i][j][9] * 4 + res_bit_RNS[i][j][8] * 8) % 11  #11
            res_RNS[i][j][4] = (res_bit_RNS[i][j][15] + res_bit_RNS[i][j][14] * 2 + res_bit_RNS[i][j][13] * 4 + res_bit_RNS[i][j][12] * 8) % 13 #13
            res_RNS[i][j][5] = (res_bit_RNS[i][j][20] + res_bit_RNS[i][j][19] * 2 + res_bit_RNS[i][j][18] * 4 + res_bit_RNS[i][j][17] * 8 + res_bit_RNS[i][j][16] * 16) % 19  #19
            res_RNS[i][j][6] = (res_bit_RNS[i][j][25] + res_bit_RNS[i][j][24] * 2 + res_bit_RNS[i][j][23] * 4 + res_bit_RNS[i][j][22] * 8 + res_bit_RNS[i][j][21] * 16) % 23  #23
            res_RNS[i][j][7] = (res_bit_RNS[i][j][30] + res_bit_RNS[i][j][29] * 2 + res_bit_RNS[i][j][28] * 4 + res_bit_RNS[i][j][27] * 8 + res_bit_RNS[i][j][26] * 16) % 29  #29
            res_RNS[i][j][8] = (res_bit_RNS[i][j][35] + res_bit_RNS[i][j][34] * 2 + res_bit_RNS[i][j][33] * 4 + res_bit_RNS[i][j][32] * 8 + res_bit_RNS[i][j][31] * 16) % 32  #32
    # RNS to number
    res = [[0] * ct_num for _ in range(8)]
    a = [0] * nRNScal
    for q in range (8):
        for i in range (ct_num):
            for j in range (nRNScal):
                a[j] = res_RNS[q][i][j+RNScalbeg]
            res[q][i] = (RNS_Combination_for_number(a,RNScal,nRNScal)) % Q
    ## 自检错 因为代码运行有问题，但这里计算机不该有错
    b = [0] * nRNSred
    RNS_corr_flag = 0
    for q in range (8):
        for i in range (ct_num):
            RNS_corr_flag = 0  
            for j in range (nRNSred):
                b[j] = res_RNS[q][i][j]
            for j in range (-8,8):    ## 因为不知道有几个正号和负号，其实边上只有正/负，中间的才会有区别
                if (((res[q][i] + Q * j) % 3 == b[0]) and ((res[q][i] + Q * j) % 5 == b[1]) and ((res[q][i] + Q * j) % 7 == b[2])):
                    RNS_corr_flag = 1
            if (RNS_corr_flag == 0):
                print("RNS error")

    ## 1 个 8 行
    res_np = np.array(res).T
    for i in range (1,8):
        res_np[:,i] = (res_np[:,i] + res_np[:,i-1]) %Q
    with open('./out/res_np.txt','w') as f:
        np.savetxt(f, res_np, fmt = '%d')
    ## 8 个 8 行的和
    # res_np_sum = (np.sum(res_np,1))% Q   ## 上面的没错，这里有错，说明超表示范围
    res_np_sum = res_np[:,7]
    with open('./out/res_np_sum.txt','w') as f:
        np.savetxt(f, res_np_sum, fmt = '%d')

    ## 正确输出（校验过）
    res_correct = np.zeros([ct_num,N])
    for i in range (ct_num):
        a = [x[i] for x in ct_list_5]
        # b = np.array(tests_pk(pk[0], [x[i] for x in u_list]))
        # res_correct[i,:] = np.array(tests_pk(pk[0], [x[i] for x in u_list])).T
        res_correct[i,:] = np.array(tests_sk(sk_c1, a)).T
    # res_correct_np = np.array(res_correct)
    with open('./out/res_correct_np.txt','w') as f:
        np.savetxt(f, res_correct, fmt = '%d')
    return


if __name__ == '__main__':
    # tests()
    # tests_FedAvg()
    # for_tester_debug_pk()
    # for_tester_debug_sk()
    tests_ct_ACF()
