import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',
    "mathtext.fontset":'cm',
    "font.serif": ['Times New Roman'],
}

rcParams.update(config)
error_dict=torch.load(r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\error_dict.pth")

cname=['$\mathrm{\it{{D_{t}}}}$','$\mathrm{\it{{D_{p}}}}$','$\mathrm{\it{{F_{p}}}}$']
noiselist=['0.02','0.04','0.06','0.08','0.1']
SNRlist=['0.02','0.04','0.06','0.08','0.1']
strategylist=['LS','Bayesian','IVIM_NEToptim','CC_PINN','MC_PINN']
namelist=['LS','Bayesian','IVIM-NET$\mathrm{{_{optim}}}$','CC-PINN','MC-PINN']
total_cname=['SNR','Strategy','D$_t$','D$_p$','F$_p$']
df=pd.DataFrame(columns=['SNR','Strategy','D$_t$','D$_p$','F$_p$'])
for i in range(len(noiselist)):
    for j in range(len(strategylist)):
        data=np.sqrt(np.mean(error_dict[strategylist[j]+'_'+str(noiselist[i])][:,[0,1,3]]**2,0))
        df_part=pd.DataFrame(np.array([SNRlist[i],namelist[j],data[0]*1000,data[1]*1000,data[2]*100]).reshape(1,5),columns=['SNR','Strategy','D$_t$','D$_p$','F$_p$'])
        df=pd.concat([df,df_part],ignore_index=True)
df['D$_t$'] = pd.to_numeric(df['D$_t$'])
df['D$_p$'] = pd.to_numeric(df['D$_p$'])
df['F$_p$'] = pd.to_numeric(df['F$_p$'])
df["SNR"] = pd.to_numeric(df["SNR"])
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(24, 8))
plt.autoscale(True)
ivimname = ['D$_t$', 'D$_p$', 'F$_p$']
plotname = ['$\mathrm{\it{{D_{t}}}}$ RMSE [$\mathrm{{10^{-3}}}$ $\mathrm{\it{{mm^{2}}}}$/s]',
            '$\mathrm{\it{{D_{p}}}}$ RMSE [$\mathrm{{10^{-3}}}$ $\mathrm{\it{{mm^{2}}}}$/s]',
            '$\mathrm{\it{{F_{p}}}}$ RMSE [%]']
snr_order = ['0.02', '0.04', '0.06', '0.08', '0.1']
ylim_list=[[0.05,0.55],[10,60],[1,27]]
ysticks = [np.arange(0.1, 0.51, 0.1), np.arange(15, 56, 10), np.arange(4, 26, 5)]
palette = sns.cubehelix_palette(6, rot=-.5, gamma=0.5, dark=0.1)[1:]
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.xticks([0.02, 0.04, 0.06, 0.08, 0.1], fontsize=36)
    plt.yticks(ysticks[i], fontsize=36)
    sns.lineplot(x="SNR", y=ivimname[i], hue="Strategy", data=df, linewidth=10, palette=palette, marker='h', markersize=25,markeredgecolor=None)
    plt.ylabel(plotname[i], fontsize=40)
    plt.ylim(ylim_list[i][0], ylim_list[i][1])
    plt.xlabel("noise level", fontsize=36)
    plt.legend().set_visible(False)
plt.subplots_adjust(wspace=0.4, hspace=0.2)
handles, labels = plt.gca().get_legend_handles_labels()
fig = plt.gcf()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=28, columnspacing=1.0, handletextpad=1,bbox_to_anchor=(0.5, -0.16), bbox_transform=plt.gcf().transFigure)
plt.savefig(r'C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\fig\RMSE.jpg' , dpi = 500, bbox_inches='tight')


cname=['$\mathrm{\it{{D_{t}}}}$','$\mathrm{\it{{D_{p}}}}$','$\mathrm{\it{{F_{p}}}}$']
noiselist=['0.02','0.04','0.06','0.08','0.1']
SNRlist=['0.02','0.04','0.06','0.08','0.1']
strategylist=['PINN','MC_IVIM_NEToptim','MC_PINN_1S','MC_PINN_3S','MC_PINN']
namelist=['PINN','MC-IVIM-NET$\mathrm{{_{optim}}}$','MC-PINN (1S)','MC-PINN (3S)','MC-PINN']
total_cname=['SNR','Strategy','D$_t$','D$_p$','F$_p$']
df=pd.DataFrame(columns=['SNR','Strategy','D$_t$','D$_p$','F$_p$'])
for i in range(len(noiselist)):
    for j in range(len(strategylist)):
        data=np.sqrt(np.mean(error_dict[strategylist[j]+'_'+str(noiselist[i])][:,[0,1,3]]**2,0))
        df_part=pd.DataFrame(np.array([SNRlist[i],namelist[j],data[0]*1000,data[1]*1000,data[2]*100]).reshape(1,5),columns=['SNR','Strategy','D$_t$','D$_p$','F$_p$'])
        df=pd.concat([df,df_part],ignore_index=True)       
df['D$_t$'] = pd.to_numeric(df['D$_t$'])
df['D$_p$'] = pd.to_numeric(df['D$_p$'])
df['F$_p$'] = pd.to_numeric(df['F$_p$'])
df["SNR"] = pd.to_numeric(df["SNR"])
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(24, 8))
plt.autoscale(True)
ivimname=['D$_t$','D$_p$','F$_p$']
plotname=['$\mathrm{\it{{D_{t}}}}$ RMSE [$\mathrm{{10^{-3}}}$ $\mathrm{\it{{mm^{2}}}}$/s]','$\mathrm{\it{{D_{p}}}}$ RMSE [$\mathrm{{10^{-3}}}$ $\mathrm{\it{{mm^{2}}}}$/s]','$\mathrm{\it{{F_{p}}}}$ RMSE [%]']
snr_order = ['0.02','0.04','0.06','0.08','0.1']
ylim_list=[[0.05,0.56],[-10,200],[0,20]]
ysticks=[np.arange(0.1,0.51,0.1),np.arange(10,181,40),np.arange(2,20,4)]
colorlist = ["#BE6C6D","#EAB67A","#D8A0C1","#7DA494","#41507A"]
palette=sns.color_palette(colorlist)
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.xticks([0.02, 0.04, 0.06, 0.08, 0.1], fontsize=36)
    plt.yticks(ysticks[i], fontsize=36)
    sns.lineplot(x="SNR", y=ivimname[i], hue="Strategy", data=df, linewidth=10, palette=palette, marker='h', markersize=25,markeredgecolor=None)
    plt.ylabel(plotname[i], fontsize=40)
    plt.ylim(ylim_list[i][0], ylim_list[i][1])
    plt.xlabel("noise level", fontsize=36)
    plt.legend().set_visible(False)
plt.subplots_adjust(wspace=0.4, hspace=0.2)
handles, labels = plt.gca().get_legend_handles_labels()
fig = plt.gcf()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=28, columnspacing=1.0, handletextpad=1,bbox_to_anchor=(0.5, -0.16), bbox_transform=plt.gcf().transFigure)
plt.savefig(r'C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\fig\RMSE_erode.jpg' , dpi = 500, bbox_inches='tight')
