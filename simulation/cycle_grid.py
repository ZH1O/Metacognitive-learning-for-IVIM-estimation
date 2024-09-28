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
lamda=0.5
consistent_item=None
multiplier_list=[1,30,1]
b_list=torch.tensor([[30,50,100,150,300,400,500]]).to(device)/1000
criterion = torch.nn.MSELoss()

data_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\data_dict.pth"
data_dict=torch.load(data_path)
train_signals,_,train_parameters=data_dict['train_'+str(noise)]
val_signals,_,val_parameters=data_dict['val_'+str(noise)]
dataloaders_dict = torch.utils.data.DataLoader(train_signals,batch_size=batch_size,shuffle=True, num_workers=0)

cycle_num_list=range(6)
translation=translation_maker(False,device,multiplier_list)
out_data=torch.zeros(0,100000,4)
for cycle_num in cycle_num_list:
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



cycle_num_list=np.arange(0,6)
camp=plt.get_cmap('Blues')
colors=camp(np.linspace(0.5,1,len(cycle_num_list)))

plt.rc('font',family='Times New Roman')
plt.figure(figsize=(50,16))
plt.rc('axes',linewidth=3)

plt.subplot(1,3,1)
ivimname=['$\mathrm{\it{{D_{t}}}}$ RMSE [$\mathrm{{10^{-3}}}$ $\mathrm{\it{{mm^{2}}}}$/s]','$\mathrm{\it{{D_{p}}}}$ RMSE [$\mathrm{{10^{-3}}}$ $\mathrm{\it{{mm^{2}}}}$/s]','$\mathrm{\it{{F_{p}}}}$ RMSE [%]']
plt.bar(cycle_num_list[1:],diff_data[1:,0],color=colors)
plt.xticks(cycle_num_list,fontsize=50)
plt.yticks(fontsize=50)
plt.xlabel('K',fontsize=50,font={"family":'Times New Roman'})
plt.ylabel(ivimname[0],fontsize=50)
plt.ylim(0.13,0.14)

plt.subplot(1,3,2)
plt.bar(cycle_num_list[1:],diff_data[1:,1],color=colors)
plt.xticks(cycle_num_list,fontsize=50)
plt.yticks(fontsize=50)
plt.xlabel('K',fontsize=50,font={"family":'Times New Roman'})
plt.ylabel(ivimname[1],fontsize=50)
plt.ylim(14,14.5)

plt.subplot(1,3,3)
plt.bar(cycle_num_list[1:],diff_data[1:,2],color=colors)
plt.xticks(cycle_num_list,fontsize=50)
plt.yticks(fontsize=50)
plt.xlabel('K',fontsize=50,font={"family":'Times New Roman'})
plt.ylabel(ivimname[2],fontsize=50,font={"family":'Times New Roman'})
plt.subplots_adjust(wspace =0.3, hspace =0.1)
plt.ylim(5.5,5.9)

plt.savefig(r'C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\fig\cycle_num.jpg', dpi = 500)
plt.show()

