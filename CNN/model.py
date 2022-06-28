
import numpy as np

def Relu(x):
    return max(0,x)
# reflects a matrix with specified number of columns horizintally to right 
def ref_y_ri(mat,num):
    mat = np.array(mat) 
    n_rows,n_cols = np.shape(mat)
    mat_e = mat
    for j in range(num):
        mat_e = np.column_stack((mat_e,mat[:,n_cols-j-1]))
    return mat_e
# reflects a matrix with specified number of rows vertically below
def ref_x_be(mat,num):
    mat = np.array(mat)
    n_rows,n_cols = np.shape(mat)
    mat_e = mat
    for i in range(num):
        mat_e = np.vstack((mat_e,mat[n_rows-i-1]))
    return mat_e
# pads an image accordingly to kernel
def pad(img,kernel):
    img = np.array(img)
    kernel = np.array(kernel)
    img_h,img_w,n_channels = np.shape(img)
    k_h,k_w,k_ch = np.shape(kernel)
    pad_img = np.zeros((img_h+k_h-1,img_w+k_w-1,k_ch))
    for k in range(n_channels):
        img_ = img[:,:,k]  
        pad_img1 = ref_x_be(img_,k_h-1)
        pad_img[:,:,k] = ref_y_ri(pad_img1,k_w-1)
    return pad_img
# convolution function for single image    
def corr(img,kernel):
    img = np.array(img)
    kernel = np.array(kernel)
    pad_img = pad(img,kernel)
    H,W,C = np.shape(img)
    k_h,k_w,k_c = np.shape(kernel)
    cor_out = np.zeros((H,W))
    for h in range(H):
        for w in range(W):
            s = 0
            for r in range(k_c):
                for y in range(k_w):
                    for t in range(k_h):
                        s = s + kernel[t,y,r]*pad_img[h+t,w+y,r]
            cor_out[h,w] = Relu(s)   
    return cor_out
# convolution layer 
# convoloves images with all the kernels and outputs the activation map
def conv_layer(img_vol,filt_kern1):
    J = corr(img_vol,filt_kern1[0])
    for i in range(1,len(filt_kern1)):
        J = np.dstack((J,corr(img_vol,filt_kern1[i])))
    return J 
# return the max element and the position 
# input must be 2x2 matrix
def max_n_index(mat):
    a = 0
    b = 0
    t = mat[0,0]
    for i in range(2):
        for j in range(2): 
            if(mat[i,j]>t):
                t = mat[i,j]
                a = i
                b = j
    return t,np.array([a,b])
# pooling of one channel #
# function also stores the position of the max elements in the unpooled activation map
def pool_2D(img):
    cache = []
    img = np.array(img)
    w,h = np.shape(img)
    out_w = int(np.ceil((w-2+1)/2))
    out_h = int(np.ceil((h-2+1)/2))
    J = np.zeros((out_w,out_h))
    for i in range(out_w):
        for j in range(out_h):
            J[i,j],ca = max_n_index(img[2*i:2*i+2,2*j:2*j+2])
            ca[0] = ca[0]+2*i
            ca[1] = ca[1]+2*j
            cache.append(ca)
    return J,np.array(cache)
# pooling of a image volume
# also stores the info of the position similar to the above function
def pool_3d(img_vol):
    cache = []
    img_vol = np.array(img_vol)
    size = 2
    stride = 2
    w,h,n_channels = np.shape(img_vol)
    out_w = int(np.ceil((w-size+1)/stride))
    out_h = int(np.ceil((h-size+1)/stride))
    J = np.zeros((out_w,out_h,n_channels))
    for d in range(n_channels):
        J[:,:,d],ca = pool_2D(img_vol[:,:,d])
        dinchak = d*np.ones((out_w*out_h,1))
        cache.append(np.hstack((ca,dinchak)))
        #cache = np.append(cache,np.hstack((ca,dinchak)),axis=0)
    return J,np.array(cache).reshape(out_w*out_h*n_channels,3)
        
## flattening function
def flat(img_vol):
    img_vol = np.array(img_vol)
    w,h,c = np.shape(img_vol)
    vec = []
    for n in range(c):
        for a in range(w):
            for b in range(h):
                vec.append(img_vol[a,b,n])
    vec = np.array(vec)
    return vec #x
# MLP of a single layer 
def layer_out(input_vec,num_nodes,weights):
    input_vec = np.array(input_vec)
    x = np.append(1,input_vec)
    out = []
    for i in range(num_nodes):
        out.append(Relu(x@weights[i]))    
    return out
# softmax function
def sft(final_out):
    return np.exp(final_out)/np.sum(np.exp(final_out))

# feed forward pass
def CNN(img_vol,kernels1,kernels2,alpha,beta):
    img_vol = img_vol.reshape(28,28,1)
    I_pad = pad(img_vol,kernels1[0])
    I1 = conv_layer(img_vol,kernels1)
    I1_mp,cache1 = pool_3d(I1)
    I1_mp_pad = pad(I1_mp,kernels2[0])
    I2 = conv_layer(I1_mp,kernels2)
    I2_mp,cache2 = pool_3d(I2)
    inp = flat(I2_mp)
    zee = layer_out(inp,49,alpha)
    see = layer_out(zee,10,beta)
    y_hat = sft(see)
    return np.array(y_hat),np.array(zee),np.array(see),np.array(inp),np.array(I2_mp),np.array(cache2).astype(int),np.array(I2),np.array(I1_mp),np.array(cache1).astype(int),np.array(I1),I1_mp_pad,I_pad
# return hidden layer output in MLP i.e z_bar
def CNN_zee(img_vol,kernels1,kernels2,alpha,beta):
    img_vol = img_vol.reshape(28,28,1)
    I1 = conv_layer(img_vol,kernels1)
    I1_mp,cache1 = pool_3d(I1)
    I2 = conv_layer(I1_mp,kernels2)
    I2_mp,cache2 = pool_3d(I2)
    inp = flat(I2_mp)
    zee = layer_out(inp,49,alpha)
    #see = layer_out(zee,10,beta)
    #y_hat = sft(see)
    return np.array(zee)
# returns input to MLP i.e x_bar
def CNN_x(img_vol,kernels1,kernels2,alpha,beta):
    img_vol = img_vol.reshape(28,28,1)
    I1 = conv_layer(img_vol,kernels1)
    I1_mp,cache1 = pool_3d(I1)
    I2 = conv_layer(I1_mp,kernels2)
    I2_mp,cache2 = pool_3d(I2)
    inp = flat(I2_mp)
    return np.array(inp)
# returns pooled activaion map after first convolutional layer
def CNN_I1_mp(img_vol,kernels1,kernels2,alpha,beta):
    img_vol = img_vol.reshape(28,28,1)
    I1 = conv_layer(img_vol,kernels1)
    I1_mp,cache1 = pool_3d(I1)
    return I1_mp


# ## BACK PROPAGATION 
##### back prop #####

# gradient of loss wrt output of output layer in MLP i.e before softmax application
def gr_s(img_vol,y,kernels1,kernels2,alpha,beta,k,y_hat):
    #y_hat = CNN(img_vol,kernels1,kernels2,alpha,beta)
    return (y_hat-y)[k]
# gradient of loss wrt output of hidden layer i.e z_m
def gr_z(img_vol,y,kernels1,kernels2,alpha,beta,m,y_hat):
    s = 0 
    for k in range(10):
        s = s + gr_s(img_vol,y,kernels1,kernels2,alpha,beta,k,y_hat)*beta[k,m]
    return s
# gradient of loss wrt MLP input i.e x_l
def gr_x(img_vol,y,kernels1,kernels2,alpha,beta,l,y_hat):
    s = 0 
    for m in range(49):
        s = s + gr_z(img_vol,y,kernels1,kernels2,alpha,beta,m,y_hat)*alpha[m,l]
    return s
# gradient of loss wrt hidden layer weights i.e beta_km
# returns a Kx(M+1) matrix 
def gr_beta(img_vol,y,kernels1,kernels2,alpha,beta,zee,y_hat):
    out = np.zeros((10,50))
    #zee = CNN_zee(img_vol,kernels1,kernels2,alpha,beta)
    #zee = np.array(zee)
    zee = np.append(1,zee)
    for k in range(10):
        for m in range(50):
            out[k,m] = gr_s(img_vol,y,kernels1,kernels2,alpha,beta,k,y_hat)*zee[m]
    return out
# gradient of loss wrt input layer weights i.e alpha_ml
# returns a Mx(L+1) matrix
def gr_alpha(img_vol,y,kernels1,kernels2,alpha,beta,zee,x_bar,y_hat):
    out = np.zeros((49,197))
    #x_bar = CNN_x(img_vol,kernels1,kernels2,alpha,beta)
    x_bar = np.append(1,x_bar)
    #zee = CNN_zee(img_vol,kernels1,kernels2,alpha,beta)
    for m in range(49):
        for l in range(197):
            val = 0
            for k in range(10):
                if (zee[m]>0):
                    val = val + gr_s(img_vol,y,kernels1,kernels2,alpha,beta,k,y_hat)*beta[k,m]*x_bar[l]
            out[m,l] = val
    return out
# gradient of loss wrt second conv layer weights.
# returns a matirx of size (4,5,5,4)
def gr_ker2(img_vol,y,kernels1,kernels2,alpha,beta,y_hat,cache2,x_bar,I1_mp_pad):
    out = np.zeros((4,5,5,4))
    f = 0
    #I1_mp = CNN_I1_mp(img_vol,kernels1,kernels2,alpha,beta)
    #img_pad = pad(I1_mp,kernels2[0])
    img_pad = I1_mp_pad
    #I2 = conv_layer(I1_mp,kernels2)
    #I2_mp,cache2 = pool_3d(I2)
    #inp = flat(I2_mp)
    inp=x_bar
    for p in range(4):
        for ch in range(4):
            for m in range(5):
                for n in range(5):
                    s = 0
                    for wulfa in range(f,f+49):
                        a = cache2[wulfa][0]
                        b = cache2[wulfa][1]
                        if (inp[wulfa]>0):
                            s = s + gr_x(img_vol,y,kernels1,kernels2,alpha,beta,wulfa,y_hat)*img_pad[int(a+m),int(b+n),ch]
                    out[p,m,n,ch] = s
        f = p*49
    return out

def gr_I1_mp(img_vol,y,kernels1,kernels2,alpha,beta,iw,ih,ip,cache2,y_hat):
    img_vol = img_vol.reshape(28,28,1)
    #I1 = conv_layer(img_vol,kernels1)
    #I1_mp,cache1 = pool_3d(I1)
    #I2 = conv_layer(I1_mp,kernels2)
    #I2_mp,cache2 = pool_3d(I2)
    s = 0
    for king in range(196):
        if (iw-cache2[king][0]>0) and (ih-cache2[king][1]>0) and (ih-cache2[king][1]<5) and (iw-cache2[king][0]<5):
            #print(iw-cache2[king][0],ih-cache2[king][1])
            s = s + gr_x(img_vol,y,kernels1,kernels2,alpha,beta,king,y_hat)*kernels2[cache2[king][2],iw-cache2[king][0],ih-cache2[king][1],ip]
    return s

def gr_w1_pmn(img_vol,y,kernels1,kernels2,alpha,beta,p,m,n,cache2,cache1,I1,y_hat,I_pad):
    s = 0
    img_vol = img_vol.reshape(28,28,1)
    img_vol_pad = I_pad
    #I1 = conv_layer(img_vol,kernels1)
    #I1_mp,cache1 = pool_3d(I1)
    cache_re = cache1.reshape(14,14,4,3)
    for w in range(14):
        for h in range(14):
            if I1[cache_re[w,h,p][0],cache_re[w,h,p][1],cache_re[w,h,p][2]]>0:
                s = s + gr_I1_mp(img_vol,y,kernels1,kernels2,alpha,beta,w,h,p,cache2,y_hat)*img_vol_pad[cache_re[w,h,p][0]+m,cache_re[w,h,p][1]+n,0]
    return s

# gradient of loss wrt first tconv layer weights.
# returns a matirx of size (4,5,5,1)
def gr_ker1(img_vol,y,kernels1,kernels2,alpha,beta,cache2,cache1,I1,y_hat,I_pad):
    out = np.zeros((4,5,5,1))
    for p in range(4):
        for m in range(5):
            for n in range(5):
                out[p,m,n,0] = gr_w1_pmn(img_vol,y,kernels1,kernels2,alpha,beta,p,m,n,cache2,cache1,I1,y_hat,I_pad)
    return out


# ##### Loading data
import torch
from torch import nn
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

train_data = torchvision.datasets.MNIST(
    root="~/Handwritten_Deep_L/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),)

kernels1 = np.random.rand(4,5,5,1)/100000
kernels2 = np.random.rand(4,5,5,4)/100000
alpha = np.random.rand(49,197)
beta = np.random.rand(10,50)
y = np.array([0,0,0,0,0,1,0,0,0,0])

y_hat,zee,see,x_bar,I2_mp,cache2,I2,I1_mp,cache1,I1,I1_mp_pad,I_pad = CNN(train_data[0][0][0],kernels1,kernels2,alpha,beta)
#print(I1.shape)
#cache_re = cache1.reshape(14,14,4,3)
#print(cache_re[0,0,1].astype(int))
#I1[0,0,0]
import timeit

start = timeit.default_timer()


r = gr_ker2(train_data[0][0][0],y,kernels1,kernels2,alpha,beta,y_hat,cache2,x_bar,I1_mp_pad)
r.shape

stop = timeit.default_timer()

print('Time: ', stop - start) 


# ### Splitting data into training and testing as specified in question
import torch
from torch import nn
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

train_data = torchvision.datasets.MNIST(
    root="~/Handwritten_Deep_L/",
    train=True,
    download=True,
)
x_train = train_data.data
y_train = train_data.targets
test_data =  torchvision.datasets.MNIST(
    root="~/Handwritten_Deep_L/",
    train=False,
    download=True,
)
x_test = test_data.data
y_test = test_data.targets

arr_idx_train = []
for i in range(10):
    t2 = np.array(np.argwhere(y_train == i))
    t2 = t2.reshape(-1)
    t1 = t2[:100]
    arr_idx_train.append(t1)
arr_idx_train = np.array(arr_idx_train).reshape(-1)
train_x = x_train[arr_idx_train]
train_y = y_train[arr_idx_train]

arr_idx_test = []
for i in range(10):
    t2 = np.array(np.argwhere(y_test == i))
    t2 = t2.reshape(-1)
    t1 = t2[:10]
    arr_idx_test.append(t1)
arr_idx_test = np.array(arr_idx_test).reshape(-1)
test_x = x_test[arr_idx_test]
test_y = y_test[arr_idx_test]
print(train_y.shape)

