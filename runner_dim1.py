import argparse
from dim1.utils import *
from dim1.models.networks import *
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
import os
from tqdm import tqdm


cuda = torch.cuda.device_count() > 0
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True}
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.style.use('seaborn-dark-palette')


parser = argparse.ArgumentParser()
parser.add_argument("--expe",type=int,help='0 = main expe, 1 = expe GAN with gradient penalty, 2 = expe n_layers')
parser.add_argument("--model",type=str,help='VAE, GAN, or SGM')
parser.add_argument("--n_runs",type=int,default=1,help="number of runs of the experiment")
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--sigma_ratio", type=float, default=0.6, help="the ratio of the geometric progression for Langevin dynamic in SGM")
parser.add_argument("--L",type=int,default=10,help="number of sigma for Langevin")
parser.add_argument("--n_step_each", type=int, default=100, help="number of iteration at each sigma in Langevin")
parser.add_argument("--step_lr",type=float,default=2e-5,help="step size of Langevin dynamic")



def get_model(model_name,n_latent=1,n_layers=3):
    if model_name=='VAE':
        return VAE(n_latent,n_layers)
    elif model_name=='GAN':
        return GAN(n_latent=n_latent,n_layers=n_layers)
    elif model_name=='SGM':
        return ScoreNet(n_latent)
    else:
        print('error: please enter VAE, GAN or SGM')


def train_model(model_name,model,Y,n_latent=1,epochs=400,batch_size=1000,normalize=False,lr=1e-4,sigma_list=[(0.6)**k for k in range(10)]):
    if model_name=='VAE':
        return train_vae(model,Y,n_latent,epochs,batch_size,normalize,lr=lr)
    elif model_name=='GAN':
        return train_gan(model,Y,epochs,batch_size,normalize,lr=lr)
    elif model_name=='SGM':
        return train_score(model,Y,sigma_list,epochs,batch_size,True,lr)
    else:
        print('error: please enter VAE, GAN or SGM')


def gen_model(model_name,model,Y,n_gen,n_latent=1,sigma_list=[(0.6)**k for k in range(10)],n_step_each=100,step_lr=2e-5):
    if model_name=='VAE':
        return generate_vae(model.decoder,Y,n_gen,n_latent)    
    elif model_name=='GAN':
        return generate_gan(model.generator,Y,n_gen,n_latent)
    elif model_name=='SGM':
        return annealed_langevin_dynamic(model,Y,sigma_list,n_latent,n_gen,n_step_each,step_lr)
    else:
        print('error: please enter VAE, GAN or SGM')


def eval_lip(model_name,model):
    if model_name=='VAE':
        return Lipschitz_constant_estimator(model.decoder)
    elif model_name=='GAN':
        return Lipschitz_constant_estimator(model.generator)
    elif model_name=='SGM':
        return Lipschitz_constant_estimator_score(model)   
    else:
        print('error: please enter VAE, GAN or SGM')  


def plot_hist(gen_data,model_name,n_runs,mu,grid,expe=0,L=None):
    fig, ax = plt.subplots(figsize=(6.4,4.8))
    pdf = [0.5*norm.pdf(x-mu)+0.5*norm.pdf(x+mu) for x in grid]
    plt.plot(grid,pdf,color='blue',alpha=0.6)
    color_dict = {'VAE':'tab:orange','GAN':'green','SGM':'tab:purple'}
    n_1, bins, patches = plt.hist(gen_data,bins=grid,color=color_dict[model_name],alpha=0.8,density=True)
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    if expe==0:
        plt.savefig('results_dim1/main_expe/histo_'+ model_name+'_mu='+str(mu)+'_run='+ str(n_runs)+'.pdf')
    else: 
        plt.savefig('results_dim1/expe_GAN_GP/histo_'+ model_name+'_L='+str(L)+'_run='+ str(n_runs)+'.pdf')        
    plt.close()
    return n_1




def main_expe(model_name,
              n_runs=10,
              epochs=400,
              batch_size=1000,
              lr=1e-4,
              sigma_list=[(0.6)**k for k in range(10)],
              n_step_each=100,
              step_lr=2e-5):
    if not os.path.isdir('results_dim1'):
        os.mkdir('results_dim1')
    if not os.path.isdir('results_dim1/main_expe'):
        os.mkdir('results_dim1/main_expe')      
    run_list_lipschitz = []
    run_list_proba = []
    run_list_bound = []
    mu_list = range(0,11)
    for k in tqdm(range(n_runs)):
        lipschitz_list = []
        proba_list = []
        bound_list = []
        for mu in tqdm(mu_list):
            model = get_model(model_name).to(device)
            Y = gaussian_mixture(mu,d=1,sigma=1,n=50000)
            train_model(model_name,model,Y,epochs=epochs,batch_size=batch_size,lr=lr,sigma_list=sigma_list)
            gen_data = gen_model(model_name,model,Y,50000,sigma_list=sigma_list,n_step_each=n_step_each,step_lr=step_lr)
            lip  = eval_lip(model_name,model)
            lipschitz_list.append(lip)
            grid = np.linspace(-20,20,400)
            n_1 = plot_hist(gen_data,model_name,k,mu,grid)
            a = int(((20 - mu/2)/40)*400)
            b = int(((20 + mu/2)/40)*400)
            pdf = [(1/2)*norm.pdf(x,loc=-mu)+(1/2)*norm.pdf(x,loc=mu) for x in grid][1:]
            if b != a:
                proba_list.append(np.mean(n_1[a:b])*(grid[b]-grid[a]))
                prob_a = np.mean(n_1[0:a])*(grid[a] - grid[0])
                alpha = norm.cdf(mu/lip + norm.ppf(prob_a)) - prob_a
                bound_list.append(alpha)
            else:
                proba_list.append(0)
                bound_list.append(0)
            del model
            torch.cuda.empty_cache()
        run_list_lipschitz.append(lipschitz_list)
        run_list_proba.append(proba_list)
        run_list_bound.append(bound_list)
    np.save('results_dim1/main_expe/main_expe_' + model_name + '.npy',np.array([run_list_lipschitz,run_list_proba,run_list_bound]))

            
            



def expe_gan_GP(n_runs=10,epochs=400,batch_size=1000,lr=1e-4,color='green'):
    if not os.path.isdir('results_dim1'):
        os.mkdir('results_dim1')
    if not os.path.isdir('results_dim1/expe_GAN_GP'):
        os.mkdir('results_dim1/expe_GAN_GP')  
    L_list = range(1,25,2)
    Y = gaussian_mixture(5,d=1,sigma=1,n=50000)  
    for k in tqdm(range(n_runs)):
        for l in tqdm(L_list):
            gan = GAN(n_latent=1,sigmoid=True).to(device)         
            train_gan(gan,Y,epochs=epochs,gradient_penalty=l) 
            gen_data = generate_gan(gan.generator,Y,50000,1)
            grid = np.linspace(-10,10,200)
            n_1 = plot_hist(gen_data,'GAN',n_runs,5,grid,1,l)
            del gan
            torch.cuda.empty_cache()


def n_layers(model_name,n_runs=10,epochs=400,batch_size=1000,lr=1e-4):
    if not os.path.isdir('results_dim1'):
        os.mkdir('results_dim1')
    if not os.path.isdir('results_dim1/expe_n_layers'):
        os.mkdir('results_dim1/expe_n_layers')  
    run_list_lipschitz = []
    for k in tqdm(range(n_runs)):
        lipschitz_list = []   
        for n_layer in range(2,7):
            model = get_model(model_name,n_layers=n_layer).to(device)
            Y = gaussian_mixture(5,d=1,sigma=1,n=50000)
            train_model(model_name,model,Y,1,epochs,batch_size,lr) 
            lip  = eval_lip(model_name,model)
            lipschitz_list.append(lip)      
            del model
            torch.cuda.empty_cache()
        run_list_lipschitz.append(lipschitz_list)
    np.save('results_dim1/expe_n_layers/expe_n_layers.npy',np.array(run_list_lipschitz)) 


if __name__ == '__main__':
    args = parser.parse_args()
    if args.expe == 0:
        sigma_list = [(args.sigma_ratio)**k for k in range(args.L)]
        main_expe(args.model,args.n_runs,args.n_epochs,args.batch_size,args.lr,sigma_list,args.n_step_each,args.step_lr)
    elif args.expe==1:
        if args.model!='GAN':
            print("error: implemented for GAN only")
        else:
            expe_gan_GP(args.n_runs,args.n_epochs,args.batch_size,args.lr)
    elif args.expe==2:
        if args.model=='SGM':
            print("error: not implemented for SGM")
        else:
            n_layers(args.model,args.n_runs,args.n_epochs,args.batch_size,args.lr)
        


    
        
    
        
