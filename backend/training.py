#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.decomposition import PCA
import torch.optim as optim
import copy

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


# In[38]:


def train(session):
    
    #<--------------Pass Parameters-------------->
    prepcaremove = session['prepca']
    noise_threshold = float(session['noise_threshold'])
    nn_widths = list(eval(session['hidden_widths']))
    hidden_depth = len(nn_widths)
    slope = float(session['slope'])
    sigmals = eval(session['L'])
    opt = session['opt']
    lr = float(session['learning_rate'])
    batch_size = int(session['batch_size'])
    epoch = int(session['training_iteration'])
    a = float(session['a'])
    model = session['model']
    n_walk = int(session['n_walk'])
    log = 200
    
    #<--------------Load data-------------->
    dir_path = os.path.dirname(os.path.abspath(__file__))
    xs = np.loadtxt(os.path.join(dir_path,'data',model).replace("\\","/"))
    n_train = xs.shape[0]
    input_dim = xs.shape[1]
    
    #<--------------Preprocessing-------------->
    # step 1: normalize each dimension
    xs = (xs - np.mean(xs, axis=0)[np.newaxis,:])/np.std(xs, axis=0)[np.newaxis,:]
    # step 2: PCA (only rotation, no scaling)
    pca = PCA()
    xs = pca.fit_transform(xs)
    # step 3: normalize or remove dimension
    remove_dim = 0
    if prepcaremove == "no":
        xs = xs/(np.std(xs,axis=0)[np.newaxis,:]+noise_threshold)
    else:
        input_dim_orig = input_dim
        input_dim = np.sum(pca.explained_variance_ratio_>noise_threshold)
        remove_dim = input_dim_orig - input_dim
        xs = xs[:,:input_dim]
        xs = xs/(np.std(xs,axis=0)[np.newaxis,:])

    nn_widths.insert(0, input_dim)
    nn_widths.append(input_dim)
        
    #<--------------Build Networks-------------->
    class den(nn.Module):
        def __init__(self):
            super(den, self).__init__()
            self.linears = nn.ModuleList([nn.Linear(nn_widths[i], nn_widths[i+1]) for i in range(hidden_depth+1)])

        def forward(self, x):
            act = nn.LeakyReLU(slope)
            self.x = x
            for i in range(hidden_depth):
                self.x = act(self.linears[i](self.x))
            self.x = self.linears[hidden_depth](self.x)
            return self.x
    
    #<--------------Training------------->
    exps = []
    losses = []

    for sigmal in sigmals:
        den_net = den()
        criterion = nn.MSELoss()
        if opt == "Adam":
            optimizer = optim.Adam(den_net.parameters(), lr = lr)
        else:
            optimizer = optim.SGD(den_net.parameters(), lr = lr)
        print("sigmal={}".format(sigmal))

        for j in range(epoch):
            den_net.train()
            optimizer.zero_grad()
            choices = np.random.choice(n_train, batch_size)
            perturb = torch.normal(0,sigmal,size=(batch_size,input_dim))
            inputs0 = torch.tensor(xs[choices], dtype=torch.float) + perturb
            outputs = den_net(inputs0)
            loss = criterion(outputs, -perturb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.data))

            if j%log == 0:
                print('Epoch:  %d | Train: %.3g' %(j, loss))

        x0 = copy.deepcopy(xs[int(n_train/2)])
        x0 = x0[np.newaxis,:]

        x0 = x0 + np.random.randn(n_walk,input_dim) * sigmal
        x0 = x0 + den_net(torch.tensor(x0,dtype=torch.float)).detach().numpy()

        pca = PCA()
        pca.fit(x0)
        svs = pca.singular_values_
        exp_ratio = svs**2/np.sum(svs**2)
        exps.append(exp_ratio)

        torch.save(den_net.state_dict(), os.path.join(dir_path,"models","%.3f"%sigmal).replace("\\","/"))

    exps = np.array(exps)
    
    #<--------------Plotting------------->
    # ERD
    def f(x,a=2):
        n = x.shape[1]
        mask = x < 1/(a*n)
        return np.sum(np.cos(np.pi/2*n*a*x)*mask,axis=1)
    
    ax1 = plt.figure(figsize=(7,5))
    exps = np.array(exps)
    for i in range(input_dim):
        plt.plot(sigmals, exps[:,i], marker="o")
    plt.xscale('log')
    plt.xlabel(r"$L$",fontsize=25)
    plt.ylabel("Explained Ratio",fontsize=25)

    ax2 = ax1.gca().twinx()
    ax2.plot(sigmals, f(exps,a=2), marker="o",color="red",linewidth=5, ls="--", markersize=15)
    plt.ylabel(r"$n_{eff}$",fontsize=25,color="red")

    plt.savefig(os.path.join(dir_path,'..','static','images','ERD.png').replace("\\","/"),bbox_inches="tight")
    plt.clf()
    # Neff histogram
    den_nets = []
    for j in range(len(sigmals)):
        sigmal = sigmals[j]
        den_net = den()
        den_net.load_state_dict(torch.load(os.path.join(dir_path,"models","%.3f"%sigmal).replace("\\","/")))
        den_nets.append(copy.deepcopy(den_net))
        
    exp_ratioss = []
    npoint = 100

    for i in range(npoint):
        if i % 20 == 0:
            print(i)
        iid = np.random.choice(n_train)
        x0 = copy.deepcopy(xs[iid])
        x0 = x0[np.newaxis,:]

        exp_ratios = []

        for j in range(len(sigmals)):

            x0 = x0 + np.random.randn(n_walk,input_dim) * sigmals[j]
            x0 = x0 + den_nets[j](torch.tensor(x0,dtype=torch.float)).detach().numpy()

            pca = PCA()
            pca.fit(x0)
            svs = pca.singular_values_
            exp_ratio = svs**2/np.sum(svs**2)
            exp_ratios.append(exp_ratio)
        exp_ratioss.append(exp_ratios)
    exp_ratioss = np.array(exp_ratioss)
    
    a = np.max(f(exp_ratioss.reshape(-1,input_dim)).reshape(npoint, len(sigmals)),axis=1)
    a = np.round(a).astype('int')
    counts = np.bincount(a)
    neff = np.argmax(counts)
    confidence = float(np.max(counts))/100.
    plt.hist(a,bins=25)
    plt.xlabel(r"$n_{eff}$",fontsize=20)
    plt.ylabel("Count",fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(dir_path,'..','static','images','Neff.png').replace("\\","/"),bbox_inches="tight")
    plt.clf()

    return neff+remove_dim, remove_dim, confidence

#<button name="train" type="submit" onClick="refreshImage('ERD','{{ url_for('static',filename='images/ERD.png') }}');refreshImage('Neff','{{ url_for('static','filename='images/Neff.png') }}">Train</button>-->