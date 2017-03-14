import numpy as np
import matplotlib.pyplot
import math
fi = [0.5,0.5]
def ini_data(Sigma,Mu1,Mu2,k,N):  
    global X  
    global Mu  
    global Expectations  
    X = np.zeros((1,N))  
    Mu = np.random.random(2)  
    Expectations = np.zeros((N,k))  
    for i in range(0,N):  
        if np.random.random(1) > 0.5:  
            X[0,i] = np.random.normal()*Sigma + Mu1  
        else:  
            X[0,i] = np.random.normal()*Sigma + Mu2 
def estep(Sigma,k,N):
    global X
    global Mu
    global Expectations
    for i in range(N):
        Denom = 0
        for j in range(k):
            Denom += (1 / (((2 * math.pi) ** 0.5) * Sigma[j])) * math.exp(- (X[0,i] - Mu[j]) ** 2 / (2 * (Sigma[j] ** 2))) * fi[j]
        for j in range(k):
            Number = (1 / (((2 * math.pi) ** 0.5) * Sigma[j])) * math.exp(- (X[0,i] - Mu[j]) ** 2 / (2 * (Sigma[j] ** 2))) * fi[j]
            Expectations[i,j] = Number / Denom
def mstep(k,N):
    global X
    global Mu
    global Expectations
    global guesssigma
    for i in range(k):
        temfi = 0
        temMu = 0
        temsigma = 0
        for j in range(N):
            temfi += Expectations[j,i]
            temMu += Expectations[j,i] * X[0,j]
            temsigma += Expectations[j,i] * ((X[0,j] - Mu[i]) ** 2)
        fi[i] = temfi / N
        Mu[i] = temMu / temfi
        guesssigma[i] = (temsigma / temfi) ** 0.5 
ini_data(6,20,40,2,400) 
error = Mu.copy()
i = 0
guesssigma = [5,6]
while True:
    i += 1
    estep(guesssigma,2,400)
    mstep(2,400)
    if abs(Mu - error).sum() < 0.001:
        break
    else:
        error = Mu.copy()
for i in range(400):
    for j in range(2):
        if Expectations[i,j] > 0.5:
            print("%f属于第%d类"%(X[0,i],j))
print(Mu)
print(fi)
print(guesssigma)

