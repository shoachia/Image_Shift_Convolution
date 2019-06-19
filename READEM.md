# Image Shift and Convolution
---
### 1. Image Shift
Compatiable for **color image** and **grayscale image**

Original Image
![](https://i.imgur.com/KAbsr0d.png)
* Periodical
```python=
if x.ndim == 2:
    color = 1
else:
    color = 3
n1 = np.shape(x)[0]
n2 = np.shape(x)[1]
xshifted = np.zeros((n1,n2,color))
irange = np.mod(np.arange(n1) + k, n1)
jrange = np.mod(np.arange(n2) + l, n2)
# firstly move upward then move rightward
xshifted = x[irange, :][:, jrange]
```
![](https://i.imgur.com/cHHB9Tp.png)

**After making the periodical shifted image, we can use this to make the following three in a very simple way.**

* Extension
```python=
m = n1 - k if k > 0 else -k-1
n = n2 - l if l > 0 else -l-1
if k != 0:
    xshifted[m::np.sign(k),:,:] = np.tile(xshifted[m-np.sign(k):m-np.sign(k)+1,:,:],(np.sign(k)*k,1,1))
if l != 0:
    xshifted[:,n::np.sign(l),:] = np.tile(xshifted[:,n-np.sign(l):n-np.sign(l)+1,:],(1,np.sign(l)*l,1))
```
![](https://i.imgur.com/V9VRqHm.png)

* Zero-pading
```python=
period = xshifted
xshifted = np.zeros_like(period)
m = n1 - k if k > 0 else -k-1
n = n2 - l if l > 0 else -l-1  
sign_k = np.sign(k) if k != 0 else 1 
sign_l = np.sign(l) if l != 0 else 1
if k == 0:
    m = n1
if l == 0:
    n = n2
if color == 3:
    xshifted[:m:sign_k,:n:sign_l,:] = period[:m:sign_k,:n:sign_l,:]
else:
    xshifted[:m:sign_k,:n:sign_l] = period[:m:sign_k,:n:sign_l]
```
![](https://i.imgur.com/SmdJzGH.png)

* Mirror
```python=
m = n1 - k if k > 0 else -k
n = n2 - l if l > 0 else -l
add_k = 1 if k < 0 else 0
add_l = 1 if l < 0 else 0
if color == 3:
    if k != 0:
        xshifted[m::np.sign(k),:,:] = xshifted[min(m,m-k):max(m,m-k) + add_k,:,:][::-np.sign(k),:,:]
    if l != 0:
        xshifted[:,n::np.sign(l),:] = xshifted[:,min(n,n-l):max(n,n-l) + add_l ,:][:,::-np.sign(l),:]
else:
    if k != 0:
        xshifted[m::np.sign(k),:] = xshifted[min(m,m-k):max(m,m-k) + add_k,:][::-np.sign(k),:]
    if l != 0:
        xshifted[:,n::np.sign(l)] = xshifted[:,min(n,n-l):max(n,n-l) + add_l][:,::-np.sign(l)]
```
![](https://i.imgur.com/NMDNZtc.png)
---
### 2. Convolution
To perform convolution, we should firstly set the kernel. In this program, we try the convolution opration on several kernels such as Gaussian, Exponential, and Box kernel.

Since naive convolution implemetation is to time consuming, using image shift can efficiently address this problem. The time can be decrease about 200 times.

* Naive Convoution
```python=
def convolve_naive(x, nu):
    n1, n2 = x.shape[:2]
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    xconv = np.zeros(x.shape)
    for i in range(s1, n1-s1):
        for j in range(s2, n2-s2):
            #kernel part
            for k in range(-s1, s1+1):
                for l in range(-s2, s2+1):
                    xconv[i][j] += x[i-k][j-l]*nu[k+s1][l+s2]
    return xconv
```
* Convolution using Image Shift
```python=
def convolve(x, nu, boundary ):
    xconv = np.zeros(x.shape)
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            xconv += nu[k+s1,l+s2]*shift(x,-k,-l,boundary)
    return xconv
```
**Besides, using the image shift can also fix the boundary problem!**








