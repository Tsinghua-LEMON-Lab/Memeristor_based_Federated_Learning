import numpy as np
import math
from enclib_BASIC import *

# RNS parameter set
RNSbase = [3,5,7,11,13,23,29,32]
nRNSbase = len(RNSbase)
RNScalbeg = 3 ## 前三个是用来校验的
RNScal = RNSbase[RNScalbeg:nRNSbase]
RNSred = RNSbase[0:RNScalbeg]
nRNScal = len(RNScal)
nRNSred = len(RNSred)
bit_len = 31 # 41

# Q = 11*13*17*19*23*29*32
# N = 64

def Matrix_Mul_Vector(M, vector, q, N):
    result = [0] * N
    for i in range (N):
        for j in range (N):
            result[i] = (result[i] + (M[i][j] * vector[j]) % q) % q 
    return result

def Matrix_Mul_Vector_nomod(M, vector, N):
    result = [0] * N
    for i in range (N):
        for j in range (N):
            result[i] = result[i] + (M[i][j] * vector[j])
    return result

def Matrix_Mul_Vector_forbit(Mall, vectorall, N, RNSbase, nRNSbase):
    bitwidth = [0] * nRNSbase
    bitwidth_all = 0
    for i in range (nRNSbase):
        bitwidth[i] = math.ceil(math.log(RNSbase[i], 2))
        bitwidth_all = bitwidth_all + bitwidth[i]
    res = [ [0] * N for _ in range(bitwidth_all) ]
    x = 0
    for i in range (nRNSbase):
        vector = vectorall[x:x+bitwidth[i]]
        for j in range (bitwidth[i]):
            for k in range (bitwidth[i]):  # 向量的每一bit都要跟矩阵相乘
                a = Matrix_Mul_Vector_nomod(Mall[x], vector[bitwidth[i]-k-1], N)
                for p in range (N):
                    a[p] = a[p] * (1<<(k))
                    res[x][p] = res[x][p] + a[p]
            x = x + 1
    return res

def Matrix_QRNS_Mul_Vector_bin(Mall, vectorall, N, RNSbase, nRNSbase):
    bitwidth = [0] * nRNSbase
    bitwidth_all = 0
    for i in range (nRNSbase):
        bitwidth[i] = math.ceil(math.log(RNSbase[i], 2))
        bitwidth_all = bitwidth_all + bitwidth[i]
    res = [ [0] * N for _ in range(bitwidth_all) ]
    x = 0
    for i in range (bitwidth_all):
        for j in range (N):  ## Mall [bits][列][行]
            for p in range (N):
                res[i][j] = res[i][j] + Mall[i][j][p] * vectorall[p]  ## 乘法是 mat行 * vec列
    return res

def Matrix_bin_Mul_Vector_QRNS(Mall, vectorall, N, RNSbase, nRNSbase):
    bitwidth = [0] * nRNSbase
    bitwidth_all = 0
    for i in range (nRNSbase):
        bitwidth[i] = math.ceil(math.log(RNSbase[i], 2))
        bitwidth_all = bitwidth_all + bitwidth[i]
    res = [ [0] * N for _ in range(bitwidth_all) ]
    for i in range (bitwidth_all):
        for j in range (N):
            for p in range (N):
                res[i][j] = res[i][j] + Mall[j][p] * vectorall[i][p]
    return res

def Vector_Mul_Vector(vectora, vectorb, q, N):
    result = [0] * N
    for i in range (N):
        result[i] = (result[i] + (vectora[i] * vectorb[i]) % q) % q 
    return result

#要定义这个运算，需要三个整数。a的模逆元素（对n取模）为b，意味着a*b mod m=1，则称a关于m的模逆为b
def gcd(a,b):
    while a!=0:
        a,b = b%a,a
    return b
#定义一个函数，参数分别为a,n，返回值为b
def Modular_Inversion(a,m):#这个扩展欧几里得算法求模逆
    if gcd(a,m)!=1:
        return None
    u1,u2,u3 = 1,0,a
    v1,v2,v3 = 0,1,m
    while v3!=0:
        q = u3//v3
        v1,v2,v3,u1,u2,u3 = (u1-q*v1),(u2-q*v2),(u3-q*v3),v1,v2,v3
    return int(u1 % m)

def dotdiv(vector, b, N, Q):
    y = [0] * N
    for i in range(N):
            y[i] = (vector[i] * Modular_Inversion(b,Q)) % Q
    return y

def RNS_Decomposition_for_number(bignum, q, n):
    smallnum = n * [0]
    for i in range (n):
        smallnum[i] = bignum % q[i]
    return smallnum

def RNS_Decomposition_for_vec(polyb):
    polyb_RNS = [ [0] * nRNSbase for _ in range(N) ]
    for i in range(N):
        polyb_RNS[i] = RNS_Decomposition_for_number(polyb[i],RNSbase,nRNSbase)
    return polyb_RNS

def RNS_Decomposition_for_mat(polya_mat):
    polya_mat_RNS = [ [ [0] * nRNSbase for _ in range(N) ] for _ in range(N) ]
    for i in range(N):
        for j in range(N):
            polya_mat_RNS[i][j] = RNS_Decomposition_for_number(polya_mat[i][j],RNSbase,nRNSbase)
    return polya_mat_RNS

def RNS_Combination_for_number(small, q, n):
    coeff = [0] * n
    bignum = 0
    q_product = 1
    for i in range(n):
        q_product = q_product * q[i]
    for i in range(n):
        q_prorns = 1
        for j in range(n):
            if(i != j):
                q_prorns = q_prorns * q[j]
        # a = Modular_Inversion(q_prorns, q[i])
        coeff[i] = q_prorns * int(Modular_Inversion(q_prorns, q[i]))
        bignum = (bignum + (coeff[i] * small[i]) % q_product) % q_product	 
    bignum = bignum % q_product
    return bignum

def RNS_Combination_for_vec(polyc_RNS):
    polyc = [0] * N
    a = [0] * nRNScal
    for i in range (N):
        for j in range (nRNScal):
            a[j] = polyc_RNS[j+RNScalbeg][i]
        polyc[i] = (RNS_Combination_for_number(a,RNScal,nRNScal)) % Q
    return polyc

    # polycbit = [0] * N
    # for i in range (N):
    #     polycbit[i] = RNS_Combination_for_number(polycbit_RNS[i], RNSbase, nRNSbase) % Q  

def random_Q(Q, N):  # coeff_modulus 随机采样
    a = np.random.randint(0, Q, size = N, dtype = np.uint64)
    return a.tolist()

def random_binary(N):   # binary 随机采样
    a = np.random.randint(0, 2, size = N, dtype = np.uint64)
    return a.tolist()

def modmul(a, b, Q, N):
    a_zero = a[:]
    b_zero = b[:]
    zero = [0] * N
    a_zero.extend(zero)
    b_zero.extend(zero)
    y = [0] * 2 * N
    # print(a)
    for i in range(2*N):
        for j in range(i + 1):
            y[i] = (y[i] + a_zero[j] * b_zero[i-j]) % Q
    ymod = [0] * N
    for i in range(N):
        ymod[i] = (y[i] - y[i+N]) % Q
    return ymod

def binexpansion(num, bitwidth):
    o_bin = bin(num)[2:] #转换为二进制，并且去掉“0b”
    out_bin = o_bin.rjust(bitwidth,'0') #原字符串右侧对齐，左侧补零，补齐后总位宽为8 #不带"0b"
    return out_bin

def decconverter(bin, bitwidth):
    o_dec = 0 #转换为二进制，并且去掉“0b”
    for i in range (bitwidth): #原字符串右侧对齐，左侧补零，补齐后总位宽为8 #不带"0b"
        o_dec = o_dec + bin[bitwidth - i - 1] * (1 << i) 
    return o_dec

def decconverter_fushu(bin, bitwidth):
    o_dec = 0 #转换为二进制，并且去掉“0b”
    for i in range (bitwidth): #原字符串右侧对齐，左侧补零，补齐后总位宽为8 #不带"0b"
        o_dec = o_dec + bin[bitwidth - i - 1] * (1 << i) 
    return o_dec

def binexp_RNS_for_number(RNS_vec, RNSbase, nRNSbase):
    bitwidth = [0] * nRNSbase
    bitwidth_all = 0
    for i in range (nRNSbase):
        bitwidth[i] = math.ceil(math.log(RNSbase[i], 2))
        bitwidth_all = bitwidth_all + bitwidth[i]
    res = []
    for i in range (nRNSbase):
        out = binexpansion(RNS_vec[i], bitwidth[i])
        out_num = [int(x) for x in out]
        res.extend(out_num)
    assert  len(res) == bitwidth_all
    return res

def binexp_RNS_for_vec(polyb_RNS):
    polyb_RNS_bit = [ [0] * bit_len for _ in range(N) ]
    for i in range (N):
        polyb_RNS_bit[i] = binexp_RNS_for_number(polyb_RNS[i],RNSbase,nRNSbase)
    # RNS binary reshape  
    polyb_RNS_bit_reshape = [ [0] * N for _ in range(bit_len) ]
    for k in range (bit_len):
        for i in range (N):
            polyb_RNS_bit_reshape[k][i] =  polyb_RNS_bit[i][k] 
    return polyb_RNS_bit_reshape

def binexp_RNS_for_mat(polya_mat_RNS):
    polya_mat_RNS_bit = [ [ [0] * bit_len for _ in range(N) ] for _ in range(N) ]
    for i in range (N):
        for j in range (N):
            polya_mat_RNS_bit[i][j] = binexp_RNS_for_number(polya_mat_RNS[i][j],RNSbase,nRNSbase)
    # RNS binary reshape  
    polya_mat_RNS_bit_reshape = [ [ [0] * N for _ in range(N) ] for _ in range(bit_len) ]
    for k in range (bit_len):
        for i in range (N):
            for j in range (N):
                polya_mat_RNS_bit_reshape[k][i][j] = polya_mat_RNS_bit[i][j][k]
    return polya_mat_RNS_bit_reshape

def deccon_RNS_for_number(RNSbit_vec, RNSbase, nRNSbase):
    bitwidth = [0] * nRNSbase
    bitwidth_all = 0
    for i in range (nRNSbase):
        bitwidth[i] = math.ceil(math.log(RNSbase[i], 2))
        bitwidth_all = bitwidth_all + bitwidth[i]
    res = []
    begin = 0
    end = 0
    for i in range (nRNSbase):
        end = begin + bitwidth[i] 
        bit_vec = RNSbit_vec[begin:end]
        out_num = (decconverter(bit_vec, bitwidth[i])) % RNSbase[i]
        res.append(out_num)
        begin = end
    assert  len(res) == nRNSbase
    return res

def deccon_RNS_for_vec(poly_bit_RNS_reshape):
    polycbit_RNS = [ [0] * nRNSbase for _ in range(N) ]
    for i in range (N):
        polycbit_RNS[i] = deccon_RNS_for_number(poly_bit_RNS_reshape[i], RNSbase, nRNSbase)
    # RESHAPE
    polycbit_RNS_reshape = [ [0] * N for _ in range(nRNSbase) ]
    for i in range (nRNSbase):
        for k in range (N):
            polycbit_RNS_reshape[i][k] = polycbit_RNS[k][i]
    return polycbit_RNS_reshape

def poly_to_mat_sk(polya):
    polya_mat = [[0] * N for _ in range(N)] 
    for i in range (N):
        for j in range (N):
            if ( i >= j ):
                polya_mat[i][j] = polya[i-j]
            if ( i < j ):
                polya_mat[i][j] = - polya[N + i - j]
    return polya_mat

def poly_to_mat_pk(polya):
    polya_mat = [[0] * N for _ in range(N)] 
    for i in range (N):
        for j in range (N):
            if ( i >= j ):
                polya_mat[i][j] = polya[i-j]
            if ( i < j ):
                polya_mat[i][j] = (- polya[N + i - j])%Q
    return polya_mat

def poly_to_mat_mod(polya):
    polya_mat = [[0] * N for _ in range(N)] 
    for i in range (N):
        for j in range (N):
            if ( i >= j ):
                polya_mat[i][j] = (polya[i-j]) % Q
            if ( i < j ):
                polya_mat[i][j] = (- polya[N + i - j]) % Q
    return polya_mat

def RNS_mul(polya, polyb):
# poly a (Q) 转为 matrix
    polya_mat = poly_to_mat_pk(polya)
# RNS Decomposition (NUM TO RNS)
    polya_mat_RNS = RNS_Decomposition_for_mat(polya_mat)
    polyb_RNS = RNS_Decomposition_for_vec(polyb) 
# binary expension (RNS TO BIT)
    polya_mat_RNS_bit = binexp_RNS_for_mat(polya_mat_RNS)
    polyb_RNS_bit = binexp_RNS_for_vec(polyb_RNS)  
# 输出为txt
    # np_grev = np.array(polya_mat_RNS_bit)
    # np_polyc = np.array(polyb_RNS_bit)
    # with open('polya_1bit.txt','w') as f:
    #     # f.write('1 bit / 74 bits: \n')
    #     np.savetxt(f, np_grev[0,:,:], fmt = '%d')
# MUL for bit
    polyc_bit_RNS = [ [0] * N for _ in range(bit_len) ]
    polyc_bit_RNS = Matrix_Mul_Vector_forbit(polya_mat_RNS_bit, polyb_RNS_bit, N, RNSbase, nRNSbase)
    # reshape
    poly_bit_RNS_reshape = [ [0] * bit_len for _ in range(N) ]
    for i in range (N):
        for k in range (bit_len):
            poly_bit_RNS_reshape[i][k] = polyc_bit_RNS[k][i]
# BIT TO RNS
    polycbit_RNS = deccon_RNS_for_vec(poly_bit_RNS_reshape)
# RNS TO NUMBER
    polycbit = RNS_Combination_for_vec(polycbit_RNS)
# 校验结果
    polyc_stupid = modmul(polya, polyb, Q, N)
    assert  polyc_stupid == polycbit
    # if (polyc_stupid == polycbit):
        # print('MVM with RNS with bit CORRECT!')
    return polycbit

## 还没改~~~
def RNS_mul_RRAM(polya, polyb):
# poly a (Q) 转为 matrix
    polya_mat = poly_to_mat_pk(polya)
# RNS Decomposition (NUM TO RNS)
    polya_mat_RNS = RNS_Decomposition_for_mat(polya_mat)
    polyb_RNS = RNS_Decomposition_for_vec(polyb) 
# binary expension (RNS TO BIT)
    polya_mat_RNS_bit = binexp_RNS_for_mat(polya_mat_RNS)
    polyb_RNS_bit = binexp_RNS_for_vec(polyb_RNS)  

# MUL for bit
    polyc_bit_RNS = [ [0] * N for _ in range(bit_len) ]
    polyc_bit_RNS = Matrix_Mul_Vector_forbit(polya_mat_RNS_bit, polyb_RNS_bit, N, RNSbase, nRNSbase)
    # reshape
    poly_bit_RNS_reshape = [ [0] * bit_len for _ in range(N) ]
    for i in range (N):
        for k in range (bit_len):
            poly_bit_RNS_reshape[i][k] = polyc_bit_RNS[k][i]
# BIT TO RNS
    polycbit_RNS = deccon_RNS_for_vec(poly_bit_RNS_reshape)
# RNS TO NUMBER
    polycbit = RNS_Combination_for_vec(polycbit_RNS)
# 校验结果
    polyc_stupid = modmul(polya, polyb, Q, N)
    assert  polyc_stupid == polycbit
    # if (polyc_stupid == polycbit):
    #     print('MVM with RNS with bit CORRECT!')
    return polycbit

# --- Tests ---
def tests_pk(polya, polyb):
# Generate random polynomial
    # polya = random_Q(Q, N)
    # polyb = random_binary(N)

# poly a 转为 matrix
    polya_mat = poly_to_mat_pk(polya)

# RNS Decomposition (NUM TO RNS)
    polya_mat_RNS = RNS_Decomposition_for_mat(polya_mat)
    # polyb_RNS = RNS_Decomposition_for_vec(polyb) 

# binary expension (RNS TO BIT)
    polya_mat_RNS_bit = binexp_RNS_for_mat(polya_mat_RNS)
    # polyb_RNS_bit = binexp_RNS_for_vec(polyb_RNS)  

# MUL for bit
    polyc_bit_RNS = [ [0] * N for _ in range(bit_len) ]
    polyc_bit_RNS = Matrix_QRNS_Mul_Vector_bin(polya_mat_RNS_bit, polyb, N, RNSbase, nRNSbase)
    # reshape
    poly_bit_RNS_reshape = [ [0] * bit_len for _ in range(N) ]
    for i in range (N):
        for k in range (bit_len):
            poly_bit_RNS_reshape[i][k] = polyc_bit_RNS[k][i]

# BIT TO RNS
    polycbit_RNS = deccon_RNS_for_vec(poly_bit_RNS_reshape)

# RNS TO NUMBER
    polycbit = RNS_Combination_for_vec(polycbit_RNS)

# 校验结果
    polyc_stupid = modmul(polya, polyb, Q, N)
    assert  polyc_stupid == polycbit
    # if (polyc_stupid == polycbit):
    #     print('MVM with RNS with bit CORRECT!')
    return polycbit


def tests_sk(polya, polyb):
# Generate random polynomial
    # polya = random_binary(N)
    # polyb = random_Q(Q, N)

# poly a 转为 matrix
    polya_mat = poly_to_mat_sk(polya)

# RNS Decomposition (NUM TO RNS)
    polyb_RNS = RNS_Decomposition_for_vec(polyb) 

# binary expension (RNS TO BIT)
    polyb_RNS_bit = binexp_RNS_for_vec(polyb_RNS)  

# MUL for bit
    polyc_bit_RNS = [ [0] * N for _ in range(bit_len) ]
    polyc_bit_RNS = Matrix_bin_Mul_Vector_QRNS(polya_mat, polyb_RNS_bit, N, RNSbase, nRNSbase)
    # reshape
    poly_bit_RNS_reshape = [ [0] * bit_len for _ in range(N) ]
    for i in range (N):
        for k in range (bit_len):
            poly_bit_RNS_reshape[i][k] = polyc_bit_RNS[k][i]

# BIT TO RNS
    polycbit_RNS = deccon_RNS_for_vec(poly_bit_RNS_reshape)

# RNS TO NUMBER
    polycbit = RNS_Combination_for_vec(polycbit_RNS)

# 校验结果
    polyc_stupid = modmul(polya, polyb, Q, N)
    assert polyc_stupid == polycbit
    # if (polyc_stupid == polycbit):
    #     print('MVM with RNS with bit CORRECT!')
    
    return polycbit

if __name__ == '__main__':
    # polya = random_Q(Q, N)
    # polyb = random_binary(N)
    # tests_pk(polya, polyb)
    # tests_sk()

    # polyb_RNS = RNS_Decomposition_for_vec(polyb) 
    # polyb_RNS_bit = binexp_RNS_for_vec(polyb_RNS)
    # print("1")


    x = 1234
    y = 5678
    xi = RNS_Decomposition_for_number(x,RNSbase,nRNSbase)
    yi = RNS_Decomposition_for_number(y,RNSbase,nRNSbase)
    zi = [0]*nRNSbase
    for i in range(nRNSbase):
        zi[i] = xi[i] * yi[i] % RNSbase[i]
    zical = [0]*nRNScal
    for i in range(nRNScal):
        zical[i] = xi[i+3] * yi[i+3] % RNScal[i]

# RNS_Combination_for_number(small, q, n)
    q_prorns = [0] * nRNScal
    ki = [0] * nRNScal
    coeff = [0] * nRNScal
    bignum = 0
    q_product = 1
    for i in range(nRNScal):
        q_product = q_product * RNScal[i]
    for i in range(nRNScal):
        q_prorns[i] = 1
        for j in range(nRNScal):
            if(i != j):
                q_prorns[i] = q_prorns[i] * RNScal[j]
        ki[i] = int(Modular_Inversion(q_prorns[i], RNScal[i]))
        coeff[i] = q_prorns[i] * ki[i]
        bignum = (bignum + (coeff[i] * zical[i]) % q_product) % q_product	 
    bignum = bignum % q_product

    zical[0] = 2
    zRNS = (RNS_Combination_for_number(zical,RNScal,nRNScal)) % Q
    z = x*y
    zmod = x*y % Q
    print(1)


    # Generate random polynomial
    # polya = random_Q(Q, N)
    # polyb = random_binary(N)
    # RNS_mul(polya, polyb)

    # polyc = random_binary(N)
    # polyd = random_Q(Q, N)
    # RNS_mul(polyc, polyd)

## 再写一个带 RRAM 噪声的