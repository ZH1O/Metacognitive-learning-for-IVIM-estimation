import numpy as np
import torch as torch
import torch.optim as optim
import sys
sys.path.append(r'C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\module')
from Model import *
from Training_model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
b_list=torch.tensor([[30,50,100,150,300,400,500]]).to(device)/1000
criterion = torch.nn.MSELoss()
num_epochs = 1000

data_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\data_dict.pth"
repeat_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\repeat_dict.pth"

data_dict=torch.load(data_path)
repeat_dict=torch.load(repeat_path)

def training_parameter(strategy):
    if strategy=='MC_PINN':
        model = initialize_model(PINN(),requires_grad=True)
        cycle_num=2
        consistent_item='weighted_parameter'
        patience=20
        batch_size =256
        multiplier_list=[1,30,1]
        lamda=0.5
        translation=translation_maker(False,device,multiplier_list)
        learning_rate=0.001
    if strategy=='MC_PINN_1S':
        model = initialize_model(PINN(),requires_grad=True)
        cycle_num=1
        consistent_item='weighted_parameter'
        patience=20
        batch_size =256
        multiplier_list=[1,30,1]
        lamda=0.5
        translation=translation_maker(False,device,multiplier_list)
        learning_rate=0.001
    if strategy=='MC_PINN_3S':
        model = initialize_model(PINN(),requires_grad=True)
        cycle_num=3
        consistent_item='weighted_parameter'
        patience=20
        batch_size =256
        multiplier_list=[1,30,1]
        lamda=0.5
        translation=translation_maker(False,device,multiplier_list)
        learning_rate=0.001
    if strategy=='Equalweight_MC_PINN':
        model = initialize_model(PINN(),requires_grad=True)
        cycle_num=2
        consistent_item='parameter'
        patience=20
        batch_size =256
        multiplier_list=[1,30,1]
        lamda=0.5
        translation=translation_maker(False,device,multiplier_list)
        learning_rate=0.001
    if strategy=='PINN':
        model = initialize_model(PINN(),requires_grad=True)
        cycle_num=0
        consistent_item='weighted_parameter'
        patience=20
        batch_size =256
        multiplier_list=[1,30,1]
        lamda=1.
        translation=translation_maker(False,device,multiplier_list)
        learning_rate=0.001
    if strategy=='CC_PINN':
        model = initialize_model(PINN(),requires_grad=True)
        cycle_num=2
        consistent_item='signal'
        patience=20
        batch_size =256
        multiplier_list=[1,30,1]
        lamda=0.5
        translation=translation_maker(False,device,multiplier_list)
        learning_rate=0.001
    if strategy=='MC_IVIM_NEToptim':
        model = initialize_model(IVIM_NEToptim(),requires_grad=True)
        consistent_item='weighted_parameter'
        cycle_num=2
        patience=10
        multiplier_list=[1,1,30,1]
        batch_size =128
        lamda=0.5
        translation=translation_maker(True,device,multiplier_list)
        learning_rate=3e-5
    if strategy=='IVIM_NEToptim':
        model = initialize_model(IVIM_NEToptim(),requires_grad=True)
        consistent_item=None
        cycle_num=0
        patience=10
        multiplier_list=[1,1,30,1]
        batch_size =128
        lamda=None
        translation=translation_maker(True,device,multiplier_list)
        learning_rate=3e-5
    return model,cycle_num,consistent_item,patience,multiplier_list,lamda,batch_size,translation,learning_rate


for strategy in ['MC_PINN','MC_PINN_1S','MC_PINN_3S','Equalweight_MC_PINN','PINN','CC_PINN','MC_IVIM_NEToptim','IVIM_NEToptim']:
    print('strategy:',strategy)
    all_residuals=np.zeros((0,4))
    for i in range(10):
        print('repeat:',i)
        noise=0.05
        train_signals,_,train_parameters=data_dict['train_'+str(noise)]
        val_signals,_,val_parameters=data_dict['val_'+str(noise)]
        model,cycle_num,consistent_item,patience,multiplier_list,lamda,batch_size,translation,learning_rate=training_parameter(strategy)
        dataloaders_dict = torch.utils.data.DataLoader(train_signals,batch_size=batch_size,shuffle=True, num_workers=0)
        model=model.to(device)
        try:
            model.train()
        except:
            pass
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)  
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
        try:
            model.eval()
        except:
            pass
        outs,parameters=test_model(model,val_signals,b_list,translation,device)
        residuals=parameters-val_parameters.numpy()
        all_residuals=np.concatenate([all_residuals,residuals],0)
    repeat_dict[strategy]=all_residuals
    torch.save(repeat_dict,repeat_path)

repeat_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\repeat_dict.pth"
repeat_dict=torch.load(repeat_path)

data_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\data_dict.pth"
data_dict=torch.load(data_path)

val_signals,_,val_parameters=data_dict['val_0.05']
mean_values=np.array(torch.mean(val_parameters,0))

m_val_parameters=val_parameters
for i in range(9):
    m_val_parameters=torch.cat((m_val_parameters,val_parameters),0)


m_val_parameters=np.array(m_val_parameters)
mean_values=np.mean(m_val_parameters,0)


for key in repeat_dict.keys(): 
    parameter=repeat_dict[key]+m_val_parameters 
    mean_values=np.mean(parameter,0) 
    std=np.std(parameter-mean_values,0) 
    print(std) 
    cv=std/mean_values 
    result=[str(np.around(cv[0],3)),str(np.around(cv[1],3)),str(np.around(cv[3],3))] 
    print(key,result)
