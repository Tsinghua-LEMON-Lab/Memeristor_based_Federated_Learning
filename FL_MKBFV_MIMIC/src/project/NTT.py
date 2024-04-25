import numpy as np

def Matrix_Mul_Vector(M, vector, q, N):
    result = [0] * N
    for i in range (N):
        for j in range (N):
            result[i] = (result[i] + (M[i][j] * vector[j]) % q) % q 
    return result

def Vector_Mul_Vector(vectora, vectorb, q, N):
    result = [0] * N
    for i in range (N):
        result[i] = (result[i] + (vectora[i] * vectorb[i]) % q) % q 
    return result

def Precalculate_G(q,g,w,N):
    G = [[0] * N for _ in range(N)] 
    for i in range (N):
        for j in range (N):
            G[i][j] = (pow(g,i*j,q) * pow(w,j,q)) % q
    return G

def Precalculate_Grev(q,g,w,N):
    G = [[0] * N for _ in range(N)] 
    for i in range (N):
        for j in range (N):
            G[i][j] = (pow(g,i*j,q) * pow(w,i,q)) % q
    return G

def yg(n):		# 这样默认求最小原根
    k=(n-1)//2
    for i in range(2,n-1):
        if pow(i,k,n)!=1:
            return i

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

def RNS_Decomposition(bignum, q, n):
    smallnum = n * [0]
    for i in range (n):
        smallnum[i] = bignum % q[i]
    return smallnum

def RNS_Combination(small, q, n):
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
        # c = q_product % q[i]
        a = Modular_Inversion(q_prorns, q[i])
        coeff[i] = q_prorns * int(a)
        bignum = (bignum + (coeff[i] * small[i]) % q_product) % q_product	 
    bignum = bignum % q_product
    return bignum

def random_Q(Q, N):  # coeff_modulus 随机采样
    a = np.random.randint(0, Q, size = N, dtype = np.uint64)
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

# --- Tests ---
def tests():
    Q = 469762049
    N = 32
# Generate 2n^th primitive root w, and root ^-1 wrev
    w = yg(Q)  
    w1 = pow(w,int((Q-1)/(2*N)),Q)
    wrev1 = Modular_Inversion(w1, Q)
# Generate n^th primitive root g, and root ^-1 grev
    g1 = pow(w,int((Q-1)/(N)),Q)
    grev1 = Modular_Inversion(g1, Q)
    G1_precal = Precalculate_G(Q, g1, w1, N)
    Grev1_precal = Precalculate_Grev(Q, grev1, wrev1, N)  
# Generate random polynomial
    # polya = N * [0]
    # polyb = N * [0]
    # for i in range (N):
    #     polya[i] = 1
    #     polyb[i] = 1
    polya = random_Q(Q, N)
    polyb = random_Q(Q, N)
# NTT
    polya_NTT = Matrix_Mul_Vector(G1_precal, polya, Q, N)
    polyb_NTT = Matrix_Mul_Vector(G1_precal, polyb, Q, N)
    polyc_INTT = Vector_Mul_Vector(polya_NTT, polyb_NTT, Q, N)
    # polyc_ = Matrix_Mul_Vector(Grev1_precal, polyc_INTT, Q, N)
    # polyc_ = dotdiv(polyc_, N, N)
# RNS
    RNSbase = [11,13,17,19,23,25,29,31,47,53,59,61,63,64]
    nRNSbase = 14
    RNS1 = [RNSbase[2]] + RNSbase[3:nRNSbase]
    # RNS2 = [RNSbase[1]] + RNSbase[3:nRNSbase]
    # RNS3 = [RNSbase[0]] + RNSbase[3:nRNSbase]
    nRNS = 12
# test
    # a = 10
    # b = 10
    # a_RNS = RNS_Decomposition(a,RNSbase,nRNSbase)
    # b_RNS = RNS_Decomposition(b,RNSbase,nRNSbase)
    # c_RNS = [0] * nRNSbase
    # for i in range (nRNSbase):
    #     c_RNS[i] = (a_RNS[i] * b_RNS[i]) % RNSbase[i]
    # c = RNS_Combination(c_RNS, RNSbase, nRNSbase)
    # print(c)

# RNS_Decomposition
    Grev1_precal_RNSbase = [ [ [0] * nRNSbase for _ in range(N) ] for _ in range(N) ]
    polyc_INTT_RNS = [ [0] * nRNSbase for _ in range(N) ]
    for i in range(N):
        for j in range(N):
            Grev1_precal_RNSbase[i][j] = RNS_Decomposition(Grev1_precal[i][j],RNSbase,nRNSbase)
        polyc_INTT_RNS[i] = RNS_Decomposition(polyc_INTT[i],RNSbase,nRNSbase)
# RNS reshape  
    Grev1_precal_RNSbase_reshape = [ [ [0] * N for _ in range(N) ] for _ in range(nRNSbase) ]
    polyc_INTT_RNS_reshape = [ [0] * N for _ in range(nRNSbase) ]
    for k in range (nRNSbase):
        for i in range (N):
            for j in range (N):
                Grev1_precal_RNSbase_reshape[k][i][j] = Grev1_precal_RNSbase[i][j][k]
            polyc_INTT_RNS_reshape[k][i] =  polyc_INTT_RNS[i][k]   

    # Grev1_precal_RNS1 = Grev1_precal_RNSbase[:][:][2] + Grev1_precal_RNSbase[:][:][3:nRNSbase]
    # Grev1_precal_RNS2 = Grev1_precal_RNSbase[:][:][1] + Grev1_precal_RNSbase[:][:][3:nRNSbase]
    # Grev1_precal_RNS3 = Grev1_precal_RNSbase[:][:][0] + Grev1_precal_RNSbase[:][:][3:nRNSbase]

# RNS debug
    debug_G = [ [0] * N for _ in range(N) ]
    debug_c =  [0] * N
    for i in range (N):
        for j in range (N):
            debug_G[i][j] = RNS_Combination(Grev1_precal_RNSbase[i][j], RNSbase,nRNSbase)
        debug_c[i] = RNS_Combination(polyc_INTT_RNS[i], RNSbase,nRNSbase)

# MUL
    polyc_RNS = [ [0] * N for _ in range(nRNSbase) ]
    for k in range (nRNSbase):
        polyc_RNS[k] = Matrix_Mul_Vector(Grev1_precal_RNSbase_reshape[k], polyc_INTT_RNS_reshape[k], RNSbase[k], N)

# RNS_Combination    
    polyc = [0] * N
    a = [0] * nRNSbase
    for i in range (N):
        for j in range (nRNSbase):
            a[j] = polyc_RNS[j][i]
        polyc[i] = (RNS_Combination(a,RNSbase,nRNSbase)) % Q
    polyc = dotdiv(polyc, N, N, Q)

# ORDINARY POLY MUL
    polyc_stupid = modmul(polya, polyb, Q, N)
    assert  polyc_stupid == polyc
    print('1')


def tests2():
    # Q = 469762049
    Q = 11*13*17*19*23*29 #2048
    N = 64 # 35
    polya = random_Q(Q, N)
    polyb = random_Q(Q, N)
    polya_mat = [[0] * N for _ in range(N)] 
    for i in range (N):
        for j in range (N):
            if ( i >= j ):
                polya_mat[i][j] = (polya[i-j]) % Q
            if ( i < j ):
                polya_mat[i][j] = (- polya[N + i - j]) % Q
    polyc = Matrix_Mul_Vector(polya_mat, polyb, Q, N)

    polyc_stupid = modmul(polya, polyb, Q, N)
    assert  polyc_stupid == polyc
    print('4')

if __name__ == '__main__':
    # tests()
    tests2()
