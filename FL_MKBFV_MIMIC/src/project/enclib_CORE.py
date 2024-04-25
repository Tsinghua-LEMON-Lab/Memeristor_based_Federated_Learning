import numpy as np
from math import floor, log
from RNSlib import *
from enclib_BASIC import *
# PATH_OUTPUT = "./out/"
PK_RRAM_error_np = np.loadtxt("./out/key storage/PK_CAL_ERR.txt", dtype = int)
PK_RRAM_error = PK_RRAM_error_np.tolist()
SK_RRAM_error_np = np.loadtxt("./out/key storage/PK_CAL_ERR.txt", dtype = int)
SK_RRAM_error = SK_RRAM_error_np.tolist()
error_range = 100

def homoadd(m0_c, m1_c):
    res_c0 = modadd(m0_c[0], m1_c[0])
    res_c1 = modadd(m0_c[1], m1_c[1])
    return res_c0,res_c1

def public_key_seedgen():
    return random_Q()

def secret_keygen():
    return random_binary()

def part_public_keygen(sk, pk_seed):
    # pk = [[0]*N for _ in range(2)]
    pk_1 = pk_seed
    e = random_normal()
    # pk_0 = modmul(sk,pk_seed)
    # pk_0 = moddotmul(modmul(sk,pk_seed), -1)
    pk_0 = modadd(moddotmul(modmul(sk,pk_seed), -1),e)
    # --- Tests ---
    # y = modadd(modmul(sk,pk_1),pk_0)
    # assert modadd(modmul(sk,pk_1),pk_0) == mod(e, Q)
    # for i in range(N):
    #     pk[0][i] = pk_0[i]
    #     pk[1][i] = pk_1[i]
    return pk_0, pk_1

def public_keymerge(pk_c1, pk_c2, pk_c3, pk_c4):
    pk0 = modadd(pk_c1[0], modadd(pk_c2[0], modadd(pk_c3[0], pk_c4[0])))
    return pk0,pk_c1[1]

def encrypt(m, pk):
    u = random_binary()
    # u = random_normal()
    e1 = random_normal()
    e2 = random_normal()
    # c0 = modmul(pk[0], u)
    # c0 = modadd(modmul(pk[0], u), e1)
    x = moddotmul(m, DELTA)
    c0 = modadd(modadd(modmul(pk[0], u), e1), x)
    # c1 = modmul(pk[1], u)
    c1 = modadd(modmul(pk[1], u), e2)
    # --- Tests ---
    # c = [[0]*N for _ in range(2)]
    # for i in range(N):
    #     c[0][i] = c0[i]
    #     c[1][i] = c1[i]
    return c0,c1

def encrypt_wRRAM(m, pk):
    u = random_binary()
    # u = random_normal()
    e1 = random_normal()
    e2 = random_normal()
    x = moddotmul(m, DELTA)
    a = RNS_mul(pk[0], u)   ## wRRAM 只需要改这个！改成读txt

    yyy = [0] * N
    xxx = PK_RRAM_error[np.random.randint(0, error_range)][:]
    err = random_Q()
    for i in range (N):
        yyy[i] = err[i] * xxx[i] + a[i]
    # a = modadd(y, a)

    # c0 = modadd(modadd(RNS_mul(pk[0], u), e1), x)
    c0 = modadd(modadd(yyy, e1), x)
    c1 = modadd(modmul(pk[1], u), e2)
    # --- Tests ---
    # c = [[0]*N for _ in range(2)]
    # for i in range(N):
    #     c[0][i] = c0[i]
    #     c[1][i] = c1[i]
    return c0,c1

def encode(num):
    pt = [0] * N
    base = 2
    if num >= 0:
        for i in range(N):
            pt[i] = int((floor(num / base ** i) % base))
    else:
        num = -num
        for i in range(N):
            pt[i] = int((floor(num / base ** i) % base))
            pt[i] = int(t - pt[i])
    return pt

def decode(pt):
    num = 0
    base = 2
    for i in range(N):
        if pt[i] < t/2:
            num = num + pt[i] * base ** i
        else:
            num = num - (t-pt[i]) * base ** i
    return num

def part_decrypt(ct, sk):
    e = random_normal()
    dct1 = RNS_mul(ct[1], sk)
    dct = modadd(dct1, e)
    return ct[0],dct

def part_decrypt_wRRAM(ct, sk):
    e = random_normal()
    dct1 = modmul(ct[1], sk)  ## 只需要改这一句话

    yyy = [0] * N
    xxx = SK_RRAM_error[np.random.randint(0, error_range)][:]
    err = random_Q()
    for i in range (N):
        yyy[i] = err[i] * xxx[i]
    dct1 = modadd(yyy, dct1)

    dct = modadd(dct1, e)
    return ct[0],dct

def decrypt_merge(dw_dct_c1, dw_dct_c2, dw_dct_c3, dw_dct_c4):
    # pt = modadd(dw_dct_c1[1], dw_dct_c2[1])
    # pt = modadd(modadd(dw_dct_c1[1], dw_dct_c2[1]), dw_dct_c3[1])
    # pt = modadd(modadd(modadd(dw_dct_c1[1], dw_dct_c2[1]), dw_dct_c3[1]), dw_dct_c4[1])
    pt = modadd(modadd(modadd(modadd(dw_dct_c1[1], dw_dct_c2[1]), dw_dct_c3[1]), dw_dct_c4[1]), dw_dct_c1[0])
    scaling = t / Q
    # noisy_p = dotmul(pt,scaling)
    # noisy_p = roundmy(dotmul(pt,scaling))
    noisy_p = mod(roundmy(dotmul(pt,scaling)), t)
    return noisy_p

# --- Tests ---
def tests():
    # --- Secret Key generation ---
    sk = secret_keygen()
    sk2 = modmul(sk, sk)
    # print(sk)
    # --- Public Key generation ---
    # pk = public_keygen(sk)
    # print(pk0)
    # print(pk1)
    # # --- Evaluate Key generation ---
    # rlks0, rlks1 = evaluate_keygen(sk)
   
    # data add encrypt data
    pt0 = [3] * 1 + [0] * (N-1)
    pt1 = [3] * 1 + [0] * (N-1)
    # m0_c = encrypt(pt0, pk)
    # m1_c = encrypt(pt1, pk)
    # c = homomul(m0_c, m1_c)
    # cre = relin(rlks0, rlks1, c)
    # p = decrypt(sk, cre)
    # print(p)

    # # encrypt data multiply encrypt data
    # num0 = 1
    # num1 = -78
    # pt0 = encode(num0)
    # pt1 = encode(num1)    
    # m0_c = encrypt(pt0, pk)
    # m1_c = encrypt(pt1, pk)
    # c = homomul(m0_c, m1_c)
    # res = relin(rlks0, rlks1, c)
    # p = decrypt(sk, res)
    # numres = decode(p)
    # print(numres)
    # # assert p[0] == (m_1 * m_2) % t

    # # encrypt data add encrypt data
    # num0 = 1
    # num1 = -78
    # pt0 = encode(num0)
    # pt1 = encode(num1)    
    # m0_c = encrypt(pt0, pk)
    # m1_c = encrypt(pt1, pk)
    # res = homoadd(m0_c, m1_c)
    # p = decrypt(sk, res)
    # numres = decode(p)
    # print(numres)
    # # assert p[0] == (m_1 * m_2) % t

    # # data multiply encrypt data
    # num0 = 4
    # num1 = -78
    # pt0 = encode(num0)
    # pt1 = encode(num1)    
    # m0_c = encrypt(pt0, pk)
    # c0 = modmul(pt1, m0_c[0])
    # c1 = modmul(pt1, m0_c[1])
    # c = [[0]*N for _ in range(2)]
    # for i in range(N):
    #     c[0][i] = c0[i]
    #     c[1][i] = c1[i]
    # p = decrypt(sk, c)
    # numres = decode(p)
    # print(numres)
    # # assert p[0] == (m_1 * m_2) % t

    # data add encrypt data
    # num0 = 4
    # num1 = -78
    # pt0 = encode(num0)
    # pt1 = encode(num1)    
    # m0_c = encrypt(pt0, pk)
    # pt1 = moddotmul(pt1, DELTA)
    # c0 = modadd(pt1, m0_c[0])
    # c1 = m0_c[1][:]
    # c = [[0]*N for _ in range(2)]
    # for i in range(N):
    #     c[0][i] = c0[i]
    #     c[1][i] = c1[i]
    # p = decrypt(sk, c)
    # numres = decode(p)
    # print(numres)
    # assert p[0] == (m_1 * m_2) % t

if __name__ == '__main__':
    tests()
