import numpy as np
import torch as torch
import torch.optim as optim
import sys
sys.path.append(r'C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\module')
from Model import *
from Training_model import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',
    "mathtext.fontset":'cm',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 1000
noise=0.05
batch_size =256
patience=20
cycle_num=2
consistent_item='weighted_parameter'
lamda=0.5
b_list=torch.tensor([[30,50,100,150,300,400,500]]).to(device)/1000
criterion = torch.nn.MSELoss()

data_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\data_dict.pth"
data_dict=torch.load(data_path)
train_signals,_,train_parameters=data_dict['train_'+str(noise)]
val_signals,_,val_parameters=data_dict['val_'+str(noise)]
dataloaders_dict = torch.utils.data.DataLoader(train_signals,batch_size=batch_size,shuffle=True, num_workers=0)

multiplier_list=np.arange(-5,5,0.1)
out_data=torch.zeros(0,100000,4)
for i in multiplier_list:
    multiplier=float(10**i)
    translation=translation_maker(False,device,[1,multiplier,1])
    model = initialize_model(PINN(),requires_grad=True)
    model=model.to(device)
    model.train() 
    optimizer = optim.Adam(model.parameters(), lr = 0.001)  

    model,loss_history= train_model(model,
                                dataloaders_dict,
                                b_list,
                                criterion,
                                optimizer,
                                num_epochs,
                                translation,
                                cycle_num,
                                consistent_item,
                                lamda,
                                device,
                                patience)
    model.eval() 
    val_signals=val_signals.to(device)
    val_parameters=val_parameters.to(device)
    X=model(val_signals) 
    outs,parameters,_=translation(X,b_list)
    parameters=parameters.reshape(1,100000,4).detach().cpu()
    out_data=torch.cat((out_data,parameters),0)


diff_data=np.sqrt(np.mean((out_data.cpu().numpy()-val_parameters.reshape(1,100000,4).cpu().numpy())**2,1))
diff_data=diff_data*np.array((1000,1000,100,100)).reshape(1,-1)

multiplier_list=np.arange(-5,5,0.1)
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(50,16))
plt.rc('axes',linewidth=3)

plt.subplot(1,3,1)
plt.axhline(y=0., color='r', linestyle=':',linewidth=5)
plt.plot(10**multiplier_list,diff_data[:,0],linewidth=7)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.xlabel('$\mathrm{τ\it{_{D_{p}}}}$',fontsize=50)
plt.ylabel('$\mathrm{\it{{D_{t}}}}$ RMSE [$\mathrm{{10^{-3}}}$ $\mathrm{\it{{mm^{2}}}}$/s]',fontsize=50)
plt.xscale("log")

plt.subplot(1,3,2)
plt.axhline(y=0., color='r', linestyle=':',linewidth=5)
plt.plot(10**multiplier_list,diff_data[:,1],linewidth=7)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.xlabel('$\mathrm{τ\it{_{D_{p}}}}$',fontsize=50)
plt.ylabel('$\mathrm{\it{{D_{p}}}}$ RMSE [$\mathrm{{10^{-3}}}$ $\mathrm{\it{{mm^{2}}}}$/s]',fontsize=50)
plt.xscale("log")

plt.subplot(1,3,3)
plt.axhline(y=0., color='r', linestyle=':',linewidth=5)
plt.plot(10**multiplier_list,diff_data[:,2],linewidth=7)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.xlabel('$\mathrm{τ\it{_{D_{p}}}}$',fontsize=50)
plt.ylabel('$\mathrm{\it{{F_{p}}}}$ RMSE [%]',fontsize=50)
plt.xscale("log")

plt.subplots_adjust(wspace =0.3, hspace =0.1)
plt.savefig(r'C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\fig\t.jpg', dpi = 500)
plt.show()

multiplier_list=np.arange(-5,5,0.1)
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(50,16))
plt.rc('axes',linewidth=10)
plt.autoscale(True)
cut_start=60
cut_end=68

plt.subplot(1,3,1)
plt.axvline(x=30., color='orange', linestyle=':',linewidth=12)
plt.plot(10**multiplier_list[cut_start:cut_end],diff_data[:,0][cut_start:cut_end],linewidth=12)
plt.xticks([12,48],fontsize=150)
plt.yticks([0.10,0.18],fontsize=150)
plt.xlim(8,52)
plt.ylim(0.09,0.19)

plt.subplot(1,3,2)
plt.axvline(x=30., color='orange', linestyle=':',linewidth=12)
plt.plot(10**multiplier_list[cut_start:cut_end],diff_data[:,1][cut_start:cut_end],linewidth=12)
plt.xticks([12,48],fontsize=150)
plt.yticks([12,17],fontsize=150)
plt.xlim(8,52)
plt.ylim(11,18)

plt.subplot(1,3,3)
plt.axvline(x=30., color='orange', linestyle=':',linewidth=12)
plt.plot(10**multiplier_list[cut_start:cut_end],diff_data[:,3][cut_start:cut_end],linewidth=12)
plt.xticks([12,48],fontsize=150)
plt.yticks([3,9],fontsize=150)
plt.xlim(8,52)
plt.ylim(2,10)
plt.subplots_adjust(wspace =0.5, hspace =0.1)

plt.savefig(r'C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\fig\mini_t.jpg', dpi = 500)
plt.show()