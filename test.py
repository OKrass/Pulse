import matplotlib.pyplot as plt
import numpy as np
import _thread
import queue
import time
import string
import random


def countC(c):
    a = np.matrix.mean(c,1)
    print(a)
    b = np.diag(a)
    d = np.power(b,-1)
    print(d)
    e = d.dot(c)-1
    return e


def findPulse(rgb):
    N = int(rgb.size / 3)
    k = 128
    B = np.matrix('6,24')
    P = np.zeros([1,N])
    for n in range (1,N-1):
        C = rgb[:,n:n+k-1]
        Cprim = countC(C)
        F = np.fft(Cprim, [], 2)
        SS = np.matrix('0,1,-1;-2,1,1')
        S = SS*F
        Z = S[1,:] + np.absolute(S[1,:])/np.absolute(S[2,:])*S[2,:]
        Zprim = Z *(np.absolute(Z)/np.absolute(np.sum(F,1)))
        Zprim[:,1:B[1]-1] = 0
        Zprim[:,B[2]+1::] = 0
        Pprim = np.real(np.ifft(Zprim,[],2))
        P[1,n:n+k-1] = P[k,n:n+k-1] + (Pprim - np.mean(Pprim))/np.std(Pprim)
    return P

N = 256

a = [random.randrange(0,255,1) for _ in range (256)]
b = [random.randrange(0,255,1) for _ in range (256)]
c = [random.randrange(0,255,1) for _ in range (256)]
rgbmatrix = np.matrix([a, b,c])
print(rgbmatrix.size)
A = findPulse(rgbmatrix)




'''
qx=queue.Queue()

def Sin(queuex):
    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(sample)


    queuex.put(x)
    y = np.sin(2 * np.pi * f * x / Fs)
    queuex.put(y)

def Get(queuex):
    if queuex.empty is 1:
        time.sleep(5)
    else:
        x = queuex.get()

        y = queuex.get()

        print (x)
        print (y)

_thread.start_new_thread(Sin, (qx,))
_thread.start_new_thread(Get, (qx,))

'''
