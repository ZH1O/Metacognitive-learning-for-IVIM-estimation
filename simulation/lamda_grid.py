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

plt.rc('font',family='Times New Roman')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 1000
noise=0.05
batch_size =256
patience=20
cycle_num=2
consistent_item='weighted_parameter'
multiplier_list=[1,30,1]
b_list=torch.tensor([[30,50,100,150,300,400,500]]).to(device)/1000
criterion = torch.nn.MSELoss()

data_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\data_dict.pth"
data_dict=torch.load(data_path)
train_signals,_,train_parameters=data_dict['train_'+str(noise)]
val_signals,_,val_parameters=data_dict['val_'+str(noise)]
dataloaders_dict = torch.utils.data.DataLoader(train_signals,batch_size=batch_size,shuffle=True, num_workers=0)


lamda_list=np.arange(0,1.01,0.01)
out_data=torch.zeros(0,100000,4)
translation=translation_maker(False,device,multiplier_list)
for lamda in lamda_list:
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




lamda_list=np.arange(0,1.01,0.01)

diff_data=np.sqrt(np.mean((out_data.cpu().numpy()-val_parameters.reshape(1,100000,4).cpu().numpy())**2,1))

diff_data=diff_data*np.array((1000,1000,100,100)).reshape(1,-1)

ivimname=['$\mathrm{\it{{D_{t}}}}$ RMSE [$\mathrm{{10^{-3}}}$ $\mathrm{\it{{mm^{2}}}}$/s]','$\mathrm{\it{{D_{p}}}}$ RMSE [$\mathrm{{10^{-3}}}$ $\mathrm{\it{{mm^{2}}}}$/s]','$\mathrm{\it{{F_{p}}}}$ RMSE [%]']

plt.figure(figsize=(64,20))
plt.rc('axes',linewidth=3)
plt.subplot(1,3,1)
plt.xticks(fontsize=80)
plt.yticks(fontsize=80)
plt.xlabel('$\it{λ}$',fontsize=100)
plt.ylabel(ivimname[0],fontsize=100)
plt.ylim(0.05,0.75)
plt.plot(lamda_list,diff_data[:,0],linewidth=8)
plt.subplot(1,3,2)
plt.xticks(fontsize=80)
plt.yticks(fontsize=80)
plt.ylim(10,45)
plt.xlabel('$\it{λ}$',fontsize=100)
plt.ylabel(ivimname[1],fontsize=100)
plt.plot(lamda_list,diff_data[:,1],linewidth=8)

plt.subplot(1,3,3)
plt.xticks(fontsize=80)
plt.yticks(fontsize=80)
plt.xlabel('$\it{λ}$',fontsize=100,font={"family":'Times New Roman'})
plt.ylabel(ivimname[2],fontsize=100,font={"family":'Times New Roman'})
plt.ylim(2,30)
plt.plot(lamda_list,diff_data[:,3],linewidth=8)

plt.subplots_adjust(wspace =0.4, hspace =0.)
plt.savefig(r'C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\fig\lamda.jpg' , dpi = 500 ,bbox_inches='tight')



multiplier_list=np.arange(-5,5,0.1)
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(60,26))
plt.rc('axes',linewidth=10)
plt.autoscale(True)
cut_start=45
cut_end=55

plt.subplot(1,3,1)
plt.axvline(x=0.5, color='orange', linestyle=':',linewidth=12)
plt.plot(lamda_list[cut_start:cut_end],diff_data[:,0][cut_start:cut_end],linewidth=12)
plt.xticks([0.45,0.55],fontsize=200)
plt.yticks([0.05,0.24],fontsize=200)
plt.xlim(0.44,0.55)
plt.ylim(0.01,0.27)
plt.yticks(fontsize=200)

plt.subplot(1,3,2)
plt.axvline(x=0.5, color='orange', linestyle=':',linewidth=12)
plt.plot(lamda_list[cut_start:cut_end],diff_data[:,1][cut_start:cut_end],linewidth=12)
plt.xticks([0.45,0.55],fontsize=200)
plt.yticks([13,16.],fontsize=200)
plt.xlim(0.44,0.55)
plt.ylim(12.4,16.6)
plt.yticks(fontsize=200)

plt.subplot(1,3,3)
plt.axvline(x=0.5, color='orange', linestyle=':',linewidth=12)
plt.plot(lamda_list[cut_start:cut_end],diff_data[:,3][cut_start:cut_end],linewidth=12)
plt.xticks([0.45,0.55],fontsize=200)
plt.yticks([4.5,6.5],fontsize=200)
plt.xlim(0.44,0.55)
plt.ylim(4.1,6.9)
plt.yticks(fontsize=200)
plt.subplots_adjust(wspace =0.5, hspace =0.1)

plt.savefig(r'C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\fig\mini_lamda.jpg', dpi = 500,bbox_inches='tight')
plt.show()

