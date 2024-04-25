# a[0] 是最低位，a[N]是最高位
import numpy as np
from math import floor, log

gau_sigma = 1000
gau_B = 10*gau_sigma
t = 32*4 # 2**10 #4  # plaintext space
Q = 11*13*23*29*32  # 469762049 # 65537*12289*40961  # 1073741824 # 65537*12289*40961
N = 64  # 1024
delta_R = pow(N, 0.5)
NOI_BU = np.floor(Q/t)/2
DELTA = np.floor(Q/t)
T = int(np.floor(pow(Q, 1/4))) # 2 # np.floor(pow(Q, 0.5))
l = int(np.floor(np.log(Q)/np.log(T)))

def modmul(a, b):
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

def mulnormal(a, b):
    a_zero = a[:]
    b_zero = b[:]
    zero = [0] * N
    a_zero.extend(zero)
    b_zero.extend(zero)
    y = [0] * 2 * N
    # print(a)
    for i in range(2*N):
        for j in range(i + 1):
            y[i] = (y[i] + a_zero[j] * b_zero[i-j])
    ymod = [0] * N
    for i in range(N):
        ymod[i] = (y[i] - y[i+N])
    return ymod

def multq(a, b):
    a_zero = a[:]
    b_zero = b[:]
    zero = [0] * N
    a_zero.extend(zero)
    b_zero.extend(zero)
    y = [0] * 2 * N
    # print(a)
    for i in range(2*N):
        for j in range(i + 1):
            y[i] = y[i] + t/Q * (a_zero[j] * b_zero[i-j])
    ymod = [0] * N
    for i in range(N):
        ymod[i] = y[i] - y[i+N]
    return ymod

def multqcomplex(a, b):
    c = mulnormal(a, b)
    c = dotmul(c, t/Q)
    return c

def modadd(a, b):
    y = [0] * N
    for i in range(N):
            y[i] = (a[i] + b[i]) % Q
    return y

def mod(a,q):
    y = [0] * N
    for i in range(N):
            y[i] = int((a[i]) % q)
    return y

def moddotmul(a, b):
    y = [0] * N
    for i in range(N):
            y[i] = int((b * a[i]) % Q)
    return y

def dotmul(a, b):
    y = [0] * N
    for i in range(N):
            y[i] = b * a[i]
    return y

def roundmy(a):
    y = [0] * N
    for i in range(N):
            y[i] = round(a[i])
    return y

def random_binary():   # binary 随机采样
    a = np.random.randint(0, 2, size = N, dtype = np.uint64)
    return a.tolist()
    # a = [1] * N
    # return a

def random_Q():  # coeff_modulus 随机采样
    a = np.random.randint(0, Q, size = N, dtype = np.uint64)
    return a.tolist()
    # a = [1000000] * N
    return a

def random_normal():  # 正态分布中采样
    a = np.int64(np.round(np.random.normal(0, gau_sigma, size = N)) )
    return a.tolist()

def base_decomp(a):
    y = [[0]*N for _ in range(l+1)]
    for i in range(l + 1):
        for j in range(N):
            ai = [0] * N
            ai[j] = int((floor(a[j] / T ** i) % T))
            y[i][j] = ai[j]   
        # result.append(np.poly1d(np.floor(polynomial / T ** i).astype(int) % T))
    return y

# --- Tests ---
def tests():
    # Q = 65537*12289*40961
    # Q2 = ( Q ** 2 + 1) * 1024
    # Q1 = Q2 % Q
    # print(Q2)
    # print(Q1)
    # Q1 = np.poly1d([Q-1] + [0]*(N-1))
    # Q2 = np.poly1d([180000] * N)\
    # Q1 = [1.5] * 2 + [0] * (N-2)
    # Q2 = [1] * 2 + [0] * (N-2)
    # Q3 = Q1[0] * Q2[0]
    # Q3 = modmul(Q1,Q2)
    # Q3 = modadd(Q3,random_binary())
    # a = [100000000]*N
    # sk = [1]*N
    # Q3 = modmul(sk,a)
    # Q3 = roundmy(Q1)
    Q1 = random_Q()
    Q2 = random_Q()
    # Q3 = mulnormal(Q1, Q2)
    # Q4 = dotmul(Q3, t/Q)
    # Q4 = roundmy(Q4)
    # Q4 = mod(Q4,Q)
    # Q5 = multq(Q1, Q2)
    # Q5 = roundmy(Q5)
    # Q5 = mod(Q5,Q)
    Q3 = base_decomp(Q1)  
    print(1)


if __name__ == '__main__':
    tests()



