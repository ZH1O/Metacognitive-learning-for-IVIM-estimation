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
parameter_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\parameter_dict.pth"
error_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\error_dict.pth"
model_path=r"C:\Users\Administrator\Desktop\code\Unity_Brain\MC_PINN\python_data\model_dict.pth"

data_dict=torch.load(data_path)

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
    parameter_dict=torch.load(parameter_path)
    error_dict=torch.load(error_path)
    model_dict=torch.load(model_path)

    for i in range(5):
        noise=(i+1)/50
        print('noise:',noise)
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

        parameter_dict[strategy+'_'+str(noise)]=parameters
        error_dict[strategy+'_'+str(noise)]=parameters-val_parameters.numpy()
        model_dict[strategy+'_'+str(noise)]=model.state_dict()

    torch.save(parameter_dict,parameter_path)
    torch.save(error_dict,error_path)
    torch.save(model_dict,model_path)

