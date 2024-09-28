import torch as torch
import time
import copy
import torch.nn as nn


relu = nn.ReLU(inplace=False)
sigmoid = nn.Sigmoid()

def translation_maker(using_sigmoid,device,factor=[1,30,1]):
    if using_sigmoid:
        def translation(code,b_list):#30% broader in each side
            s0=sigmoid(code[:,0]).reshape(-1,1)*0.96+0.52# 0.7-1.3
            Dslow=sigmoid(code[:,1]).reshape(-1,1)*4.8-0.9# 0-3
            Dfast=sigmoid(code[:,2]).reshape(-1,1)*160-30# 0-100
            Fslow=sigmoid(code[:,3]).reshape(-1,1)*1.6-0.3# 0.-1
            Ffast=relu(1-Fslow)  
            signals=(Fslow*torch.exp(-b_list*Dslow)+Ffast*torch.exp(-b_list*Dfast))*s0
            parameters=torch.cat((Dslow/1000,Dfast/1000,Fslow,Ffast), dim=1)
            weighted_parameters=torch.cat((s0/factor[0],Dslow/factor[1],Dfast/factor[2],Fslow/factor[3],Ffast/factor[3]), dim=1)
            return signals,parameters,weighted_parameters
    else:
        factor=torch.tensor(factor).reshape(1,-1).to(device)
        def translation(code,b_list):
            outs=code*factor
            Dslow=outs[:,0].reshape(-1,1)
            Dfast=outs[:,1].reshape(-1,1)   
            Fslow=outs[:,2].reshape(-1,1) 
            Ffast=relu(1-Fslow)  
            signals=Fslow*torch.exp(-b_list*Dslow)+Ffast*torch.exp(-b_list*Dfast)
            parameters=torch.cat((Dslow/1000,Dfast/1000,Fslow,Ffast), dim=1)
            return signals,parameters,code
    return translation



def train_model(model,dataloaders,b_list,criterion,optimizer,num_epochs,translation,cycle_num,consistent_item,lamda,device,patience):
    since = time.time()
    loss_history = [] 
    min_loss = 100.
    bad_epochs=0.
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)
        running_loss = 0. 
        for inputs in dataloaders:  
            inputs=inputs.to(device)
            cycle_X_list=[]
            outs_parameter_weightedparameter_list=[]
            with torch.autograd.set_grad_enabled(True):
                cycle_X_list.append(model(inputs))
                outs_parameter_weightedparameter_list.append(translation(cycle_X_list[0],b_list))
                if cycle_num!=0:
                    loss_mc=0.
                    for i in range(cycle_num):
                        cycle_X_list.append(model(outs_parameter_weightedparameter_list[i][0]))
                        outs_parameter_weightedparameter_list.append(translation(cycle_X_list[i+1],b_list))
                        if consistent_item=='weighted_parameter':
                            loss_mc+=criterion(outs_parameter_weightedparameter_list[0][2],outs_parameter_weightedparameter_list[i+1][2])
                        if consistent_item=='parameter':
                            loss_mc+=criterion(outs_parameter_weightedparameter_list[0][1],outs_parameter_weightedparameter_list[i+1][1])
                        if consistent_item=='signal':
                            loss_mc+=criterion(outs_parameter_weightedparameter_list[0][0],outs_parameter_weightedparameter_list[i+1][0])
                            
                loss_fit=criterion(inputs,outs_parameter_weightedparameter_list[0][0])
                if cycle_num!=0:
                    loss=lamda*loss_fit+(1-lamda)*loss_mc*(1/cycle_num)
                else:
                    loss=loss_fit
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*inputs.shape[0]
            
        epoch_loss = running_loss/dataloaders.dataset.shape[0]
        

        print("Loss for updata: {} ,number of bad_epochs:{}".format(epoch_loss,bad_epochs))
        loss_history.append(epoch_loss) 
        if epoch_loss<=min_loss:
            min_loss=epoch_loss
            best_model = copy.deepcopy(model.state_dict())
            bad_epochs=0.
        else:
            bad_epochs+=1.
            if bad_epochs == patience:
                print('Early stopping')
                break
    model.load_state_dict(best_model)
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    return model,loss_history

def test_model(model,test_data,b_list,translation,device):
    dataloaders = torch.utils.data.DataLoader(test_data,batch_size=256,shuffle=False, num_workers=0)
    outs=torch.zeros(0,test_data.shape[-1])
    parameters=torch.zeros(0,4)
    for inputs in dataloaders:  
        inputs=inputs.to(device)
        X=model(inputs) 
        out,parameter,_=translation(X,b_list)
        outs=torch.cat((outs,out.detach().cpu()),0)
        parameters=torch.cat((parameters,parameter.detach().cpu()),0)
    return outs.numpy(),parameters.numpy()
