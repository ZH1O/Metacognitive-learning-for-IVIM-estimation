import torch
import sys
sys.path.append(r'C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\module')
from Fitting import *
import time

data_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\data_dict.pth"
parameter_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\parameter_dict.pth"
error_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\error_dict.pth"

data_dict=torch.load(data_path)
parameter_dict=torch.load(parameter_path)
error_dict=torch.load(error_path)

b_list=np.array([30,50,100,150,300,400,500])/1000
parameters0=parameters0_maker()
print(len(parameters0))

for i in range(5):
    since = time.time()
    noise=(i+1)/50
    print(noise)
    val_signals,_,val_parameters=data_dict['val_'+str(noise)]
    val_signals=val_signals.numpy()
    val_parameters=val_parameters.numpy()
    ls_parameters=LS(val_signals,parameters0,b_list)
    parameter_dict['LS_'+str(noise)]=ls_parameters
    error_dict['LS_'+str(noise)]=ls_parameters-val_parameters
    
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))

torch.save(parameter_dict,parameter_path)
torch.save(error_dict,error_path)

parameter_dict=torch.load(parameter_path)
error_dict=torch.load(error_path)

for i in range(5):
    since = time.time()
    noise=(i+1)/50
    print(noise)
    val_signals,_,val_parameters=data_dict['val_'+str(noise)]
    val_signals=val_signals.numpy()
    val_parameters=val_parameters.numpy()
    
    ls_parameters=parameter_dict['LS_'+str(noise)]
    Dslow0=ls_parameters[:,0].flatten()*1000
    Dfast0=ls_parameters[:,1].flatten()*1000
    Fslow0=ls_parameters[:,2].flatten()
    neg_log_prior_fun=empirical_neg_log_prior(Dslow0, Dfast0, Fslow0)
    bayesian_parameters=Bayesian(val_signals,neg_log_prior_fun,[[1,30,0.80]],b_list)
    parameter_dict['Bayesian_'+str(noise)]=bayesian_parameters
    error_dict['Bayesian_'+str(noise)]=bayesian_parameters-val_parameters
    
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))

torch.save(parameter_dict,parameter_path)
torch.save(error_dict,error_path)

