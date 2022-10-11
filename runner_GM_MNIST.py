import argparse
from MNIST.classifier import *
from MNIST.models.push_forward import *
from MNIST.models.UNet import UNet_simple
from MNIST.utils import *
import matplotlib
import matplotlib.pyplot as plt
import sys
import os 
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


cuda = torch.cuda.device_count() > 0
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True}



parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,help='VAE, GAN, SGM, or data')
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--sigma_ratio", type=float, default=0.6, help="the ratio of the geometric progression for Langevin dynamic in SGM")
parser.add_argument("--L",type=int,default=10,help="number of sigma for Langevin")
parser.add_argument("--n_step_each", type=int, default=100, help="number of iteration at each sigma in Langevin")
parser.add_argument("--step_lr",type=float,default=2e-5,help="step size of Langevin dynamic")
parser.add_argument("--digita",type=int,default=3,help='the first digit')
parser.add_argument("--digitb",type=int,default=7,help='the second digit')
parser.add_argument("--batch_gen",type=int,default=500,help='size of batch for generation')
parser.add_argument("--n_gen",type=int,default=5000,help='number of generated data for histograms')
parser.add_argument("--n_latent",type=int,default=784,help='dimension of the latent space for the VAE and the GAN')
parser.add_argument("--n_feature",type=int,default=32,help='number of channels for VAE and GAN (x2 for the SGM)')


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


def histo_projection(og_img,data,classifier,a=3,b=7):
    unit_vec = (torch.flatten(og_img[1]) - torch.flatten(og_img[0]))/torch.norm(og_img[1]-og_img[0])  
    prod_list_a = []
    prod_list_b = []
    prod_list_c = []
    for x in data:
        if isinstance(x,(list,tuple)) and x[1] == a:
            prod_list_a.append(torch.dot(torch.flatten(x[0]),unit_vec).item())
        elif isinstance(x,(list,tuple)) and x[1] == b:
            prod_list_b.append(torch.dot(torch.flatten(x[0]),unit_vec).item())
        else:
            c = torch.argmax(torch.softmax(classifier(x),dim=1),dim=1).item()

            if c == a:
                prod_list_a.append(torch.dot(torch.flatten(x),unit_vec).item())
            elif c == b:
                prod_list_b.append(torch.dot(torch.flatten(x),unit_vec).item())    
            else:
                prod_list_c.append(torch.dot(torch.flatten(x),unit_vec).item())
    return prod_list_a,prod_list_b,prod_list_c


def plot_hist(a,b,c,model,digita,digitb):
    fig, ax = plt.subplots()
    plt.style.use('seaborn-dark-palette')
    if len(c)==0:
        plt.hist([a,b],bins=400,density=True,color=['tab:blue','tab:green'],stacked=True) 
    else:
        plt.hist([a,b,c],bins=400,density=True,color=['tab:blue','tab:green','tab:red'],stacked=True) 
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) 
    plt.tight_layout()
    plt.savefig('results_MNIST_GM/histogram_GM_'+str(digita)+'_'+str(digitb)+'_'+model+'.pdf')
    plt.close()


def get_model(model_name,n_latent=784,n_feature=32):
    if model_name=='VAE':
        return VAE(n_latent,n_feature)
    elif model_name=='GAN':
        return GAN(n_latent,n_feature,sigmoid=False,conv_disc=False)
    elif model_name=='SGM':
        return UNet_simple(False,2*n_feature)
    else:
        print('error: please enter VAE, GAN or SGM')


def train_model(model_name,model,train_dataset,epochs=100,batch_size=128,lr=2e-4,sigma_list=[(0.6)**k for k in range(10)]):
    if model_name=='VAE':
        return train_vae(model,train_dataset,epochs,batch_size,lr)
    elif model_name=='GAN':
        return train_hinge(model,train_dataset,epochs,batch_size,lr)
    elif model_name=='SGM':
        return train_score(model,train_dataset,sigma_list,epochs,batch_size,lr)
    else:
        print('error: please enter VAE, GAN or SGM')


def gen_model(model_name,model,n_gen,n_latent=784,sigma_list=[(0.6)**k for k in range(10)],n_step_each=100,step_lr=2e-5):
    if model_name=='VAE':
        return generate_vae(model.decoder,n_gen,n_latent)    
    elif model_name=='GAN':
        return generate_gan(model.generator,n_gen,n_latent)
    elif model_name=='SGM':
        return annealed_langevin_dynamic(model,sigma_list,n_gen,n_step_each,step_lr)
    else:
        print('error: please enter VAE, GAN or SGM')



if __name__ == '__main__':
    if not os.path.isdir('results_MNIST_GM'):
        os.mkdir('results_MNIST_GM')
    args = parser.parse_args()
    new_train_data, og_img = synthetic_gm(train_data,a=args.digita,b=args.digitb)
    classifier = MnistClassifier()
    if args.model == 'data':
        a,b,c = histo_projection(og_img,new_train_data,classifier,args.digita,args.digitb)
        plot_hist(a,b,c,args.model,args.digita,args.digitb)
        sys.exit(0)
    if os.path.isfile('MNIST/classifier.pt'):
        classifier.load_state_dict(torch.load('MNIST/classifier.pt'))
    else:
        print('training classifier on MNIST')
        train_classifier(classifier,train_data,epochs=10,batch_size=args.batch_size,lr=args.lr)
        print('accuracy on test set: ' + str(test_classifier(classifier,test_data,batch_size=args.batch_size)))
        torch.save(classifier.state_dict(),'MNIST/classifier.pt')
    sigma_list = [(args.sigma_ratio)**k for k in range(args.L)]
    model = get_model(args.model,args.n_latent,args.n_feature).to(device)
    if not os.path.isfile('results_MNIST_GM/gen_data_MNIST_GM_'+str(args.digita)+'_'+str(args.digitb)+'_'+args.model+'.pt'):
        if not os.path.isfile('results_MNIST_GM/MNIST_GM_'+str(args.digita)+'_'+str(args.digitb)+'_'+args.model+'.pt'):
            print('training '+ args.model)
            train_model(args.model,model,new_train_data,args.n_epochs,args.batch_size,args.lr,sigma_list)
            torch.save(model.state_dict(),'results_MNIST_GM/MNIST_GM_'+str(args.digita)+'_'+str(args.digitb)+'_'+args.model+'.pt')
        else:
            model.load_state_dict(torch.load('results_MNIST_GM/MNIST_GM_'+str(args.digita)+'_'+str(args.digitb)+'_'+args.model+'.pt'))
        gen_data = []
        for k in range(int(args.n_gen/args.batch_gen)):
            gen_data.append(gen_model(args.model,model,args.batch_gen,args.n_latent,sigma_list,args.n_step_each,args.step_lr))
        gen_data = torch.cat(gen_data,dim=0)
        torch.save(gen_data,'results_MNIST_GM/gen_data_MNIST_GM_'+str(args.digita)+'_'+str(args.digitb)+'_'+args.model+'.pt')
    else:
        gen_data = torch.load('results_MNIST_GM/gen_data_MNIST_GM_'+str(args.digita)+'_'+str(args.digitb)+'_'+args.model+'.pt')
    a,b,c = histo_projection(og_img,gen_data,classifier,args.digita,args.digitb)
    plot_hist(a,b,c,args.model,args.digita,args.digitb)
    sys.exit(0)
        
