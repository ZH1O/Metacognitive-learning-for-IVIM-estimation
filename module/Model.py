import torch as torch
import torch.nn as nn

def init_parameter(model):
    if isinstance(model, nn.Linear):
        if model.weight is not None:
            torch.nn.init.kaiming_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.normal_(model.bias.data)
    elif isinstance(model, nn.BatchNorm1d):
        if model.weight is not None:
            torch.nn.init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            torch.nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        if model.weight is not None:
            torch.nn.init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            torch.nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm3d):
        if model.weight is not None:
            torch.nn.init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            torch.nn.init.constant_(model.bias.data, 0)
    else:
        pass 
            
def initialize_model(model,requires_grad):
    for param in model.parameters():
        init_parameter(param.data)
    return model


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.NET = nn.Sequential()
        self.NET.add_module("Linear1",nn.Linear(7, 32))
        self.NET.add_module("elu1",nn.ELU())
        self.NET.add_module("Linear2",nn.Linear(32, 32))
        self.NET.add_module("elu2",nn.ELU())
        self.NET.add_module("Linear3",nn.Linear(32, 32))
        self.NET.add_module("elu3",nn.ELU())
        self.NET.add_module("Linear4",nn.Linear(32, 32))
        self.NET.add_module("elu4",nn.ELU())
        self.NET.add_module("Linear5",nn.Linear(32, 32))
        self.NET.add_module("elu5",nn.ELU())
        self.NET.add_module("Linear6",nn.Linear(32, 3))
    def forward(self, x):
        return torch.abs(self.NET(x))


class IVIM_NEToptim(nn.Module):
    def __init__(self):
        super(IVIM_NEToptim, self).__init__()
        self.model1=nn.Sequential()
        self.model2=nn.Sequential()
        self.model3=nn.Sequential()
        self.model4=nn.Sequential()
        
        self.model1.append(nn.Linear(7, 7))
        self.model1.append(nn.BatchNorm1d(7))
        self.model1.append(nn.ELU())
        self.model2.append(nn.Linear(7, 7))
        self.model2.append(nn.BatchNorm1d(7))
        self.model2.append(nn.ELU())
        self.model3.append(nn.Linear(7, 7))
        self.model3.append(nn.BatchNorm1d(7))
        self.model3.append(nn.ELU())
        self.model4.append(nn.Linear(7, 7))
        self.model4.append(nn.BatchNorm1d(7))
        self.model4.append(nn.ELU())
        
        for i in range(2):
            self.model1.append(nn.Linear(7, 7))
            self.model1.append(nn.BatchNorm1d(7))
            self.model1.append(nn.ELU())
            self.model1.append(nn.Dropout(0.1))
            
            self.model2.append(nn.Linear(7, 7))
            self.model2.append(nn.BatchNorm1d(7))
            self.model2.append(nn.ELU())
            self.model2.append(nn.Dropout(0.1))
            
            self.model3.append(nn.Linear(7, 7))
            self.model3.append(nn.BatchNorm1d(7))
            self.model3.append(nn.ELU())
            self.model3.append(nn.Dropout(0.1))
            
            self.model4.append(nn.Linear(7, 7))
            self.model4.append(nn.BatchNorm1d(7))
            self.model4.append(nn.ELU())
            self.model4.append(nn.Dropout(0.1))
            
        self.model1.append(nn.Linear(7, 1)) 
        self.model2.append(nn.Linear(7, 1)) 
        self.model3.append(nn.Linear(7, 1))
        self.model4.append(nn.Linear(7, 1))

    def forward(self, x):
        s0=self.model1(x)
        dslow=self.model2(x)
        dfast=self.model3(x)
        fslow=self.model4(x)
        return torch.cat((s0,dslow,dfast,fslow),1)