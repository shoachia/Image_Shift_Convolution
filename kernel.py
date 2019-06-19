import numpy as np
def kernel(name, tau=1, eps=1e-3):
    if name == 'gaussian':
        s1 = 0
        while True:
            if np.exp(-(s1**2)/(2*tau)) < eps:
                break
            s1 += 1
        s1 = s1-1
        s2 = s1
        i = np.arange(-s1,s1+1) #-3 ~ 3
        j = np.arange(-s2,s2+1) #-3 ~ 3 
        ii, jj = np.meshgrid(i, j, sparse=True,indexing='ij')
        nu = np.exp(-(ii**2 + jj**2) / (2*tau**2))
        nu[nu < eps] = 0
        nu /= nu.sum()
    elif name == 'exponential':
        tau = 3
        s1 = 0
        while True:
            if np.exp(-(s1)/(tau)) < eps:
                break
            s1 += 1
        s1 = s1-1
        s2 = s1
        i = np.arange(-s1,s1+1) #-20 ~ 20
        j = np.arange(-s2,s2+1) #-20 ~ 20
        ii, jj = np.meshgrid(i, j, sparse=True,indexing='ij')
        nu = np.exp(-(np.sqrt(ii**2+jj**2))/tau)
        nu[nu < eps] = 0
        nu /= nu.sum()
    elif name == 'box':
        s1 = tau
        s2 = tau
        i = np.arange(-s1,s1+1) #-1 ~ 1
        j = np.arange(-s2,s2+1) #-1 ~ 1
        ii, jj = np.meshgrid(i, j, sparse=True,indexing='ij')
        nu = np.exp(0*(ii+jj))
        nu[nu < eps] = 0
        nu /= nu.sum()
    else:
        raise ValueError('invalid kernel')
    return nu
