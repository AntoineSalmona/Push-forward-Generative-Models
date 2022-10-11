import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import ot
import multiprocessing

"""
Pytorch transcription of https://github.com/mducoffe/Learning-Wasserstein-Embeddings
"""



train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
    
)


test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)



n_proc = multiprocessing.cpu_count()

def run_emd(data,n_samples,file='MNIST/DWE/emd_mnist.npy'):
    idx_1 = np.random.randint(0,len(data),n_samples)
    idx_2 = np.random.randint(0,len(data),n_samples)
    xx,yy = np.meshgrid(np.arange(28),np.arange(28))
    xy = np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
    M = ot.dist(xy, xy)
    
    def compute_emd(i):
        x = data[idx_1[i]][0]
        y = data[idx_2[i]][0]
        x = x.double().numpy()
        y = y.double().numpy()
        x /= np.sum(x)
        y /= np.sum(y)
        return np.float32(ot.emd2(x.reshape(784),y.reshape(784),M))
    
    loop = tqdm(range(n_samples))
    dist_list = np.array(ot.utils.parmap(compute_emd,loop,n_proc))
    save = np.array([idx_1,idx_2,dist_list])
    np.save(file,save)


