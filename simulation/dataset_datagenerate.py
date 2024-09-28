import torch as torch

total_num = 100000
b_list=torch.tensor([[0,30,50,100,150,300,400,500]])
ivim_b_index=range(8)
ivim_bnum=len(ivim_b_index)

def ivim(Dslow,Dfast,Fslow,Ffast):
    return Fslow*torch.exp(-b_list[:,ivim_b_index]*Dslow)+Ffast*torch.exp(-b_list[:,ivim_b_index]*Dfast)

def signals_maker(size,noise_min,noise_max):    
    Dslow = torch.Tensor(size,1).uniform_(0.0005, 0.001)
    Dfast = torch.Tensor(size,1).uniform_(0.01, 0.06)
    Fslow = torch.Tensor(size,1).uniform_(0.75, 0.95)
    Ffast = 1-Fslow
    
    ivim_no_noise = ivim(Dslow,Dfast,Fslow,Ffast)
    ivim_noise1 = torch.normal(mean=0.0,std=torch.Tensor(size,1).uniform_(noise_min, noise_max)*torch.ones(1,ivim_bnum).float())
    ivim_noise2 = torch.normal(mean=0.0,std=torch.Tensor(size,1).uniform_(noise_min, noise_max)*torch.ones(1,ivim_bnum).float())

    ivim_signals = torch.sqrt((ivim_no_noise+ivim_noise1)**2+ivim_noise2**2)
    
    ivim_signals = ivim_signals[:,1:]/ivim_signals[:,0].reshape(-1,1)
    ivim_no_noise = ivim_no_noise[:,1:]/ivim_no_noise[:,0].reshape(-1,1)

    ivim_parameters=torch.cat((Dslow,Dfast,Fslow,Ffast), dim=1)
    return [ivim_signals,ivim_no_noise,ivim_parameters]

data_dict={}
phase_list=['train','val']
for phase in phase_list:
    for i in range(10):
        noise=(i+1)/100
        key=phase+'_'+str(noise)
        data_dict[key]=signals_maker(total_num,noise_min=noise,noise_max=noise)


torch.save(data_dict,r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\data_dict.pth")
error_dict={}
torch.save(error_dict,r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\error_dict.pth")
parameter_dict={}
torch.save(parameter_dict,r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\parameter_dict.pth")
model_dict={}
torch.save(model_dict,r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\model_dict.pth")

