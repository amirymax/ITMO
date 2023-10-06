import numpy as np

def Chebyshev_nodes(f,N,a,b):
    x = []
    for i in range(N):
        x.append((a+b)/2 + (b-a)/2*np.cos((2*i+1)*np.pi/(2*N)))
    fvals = []
    for i in range(N):
        fvals.append(f(x[i]))
    return x,fvals

def Lagrange(x, fvals, xint):
    n = len(x)
    L = 0.
    for j in range(n):
        basic_polynomial= 1.
        for i in range(n):
            if i != j:
                basic_polynomial *= (xint - x[i])/(x[j] - x[i])
        L += fvals[j]*basic_polynomial
    return L