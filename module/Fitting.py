import numpy as np
from scipy.optimize import curve_fit,minimize
from scipy import stats

def parameters0_maker():
    f_array=[]
    for x0 in range(0,31,15):
        for x1 in range(10,101,30):
            for x2 in range(10,101,30):
                dslow=x0/10
                dfast=x1
                fslow=x2/100
                f_array.append([dslow,dfast,fslow])
    return f_array

def ivim(code,b_list):
    Dslow=code[:,0].reshape(-1,1)
    Dfast=code[:,1].reshape(-1,1)   
    Fslow=code[:,2].reshape(-1,1) 
    Ffast=1-Fslow
    signals=Fslow*np.exp(-b_list*Dslow)+Ffast*np.exp(-b_list*Dfast)
    parameters=np.concatenate((Dslow,Dfast,Fslow,Ffast), axis=1)
    return signals,parameters

def equation(b, Dslow,Dfast,Fslow):
    return Fslow*np.exp(-b*Dslow) + (1-Fslow)*np.exp(-b*Dfast)

def order(Dslow,Dfast,Fslow):
    if Dfast < Dslow:
        Dfast, Dslow = Dslow, Dfast
        Fslow = 1-Fslow   
    return np.array([Dslow,Dfast,Fslow,1-Fslow])

def fit_least_squares(X,Y,p0):
    try:
        bounds = ([0., 0, 0.], [3, 100., 1.])
        params, _ = curve_fit(equation, X, Y, p0=p0, bounds=bounds)
        dslow,dfast,fslow = params[0], params[1], params[2]
        return order(dslow,dfast,fslow)
    except:
        return np.array([0,0,0,0])
        
def LS(signals,parameters0,b_list):
    x,y=signals.shape
    ls_parameters=np.zeros((x, 4)) 
    for i in range(x):
        Y=signals[i,:]
        if Y[0]<1e-2:
            ls_parameters[i,:]=np.array([0.,0.,0.,0.])
        else:
            Yt=Y.reshape(1,-1)
            loss_min=1e8
            for j in range(len(parameters0)): 
                p=parameters0[j]
                temp_parameters=fit_least_squares(b_list,Y,p)
                temp_parameters=np.array(temp_parameters).reshape(1,-1)
                temp_signals,_=ivim(temp_parameters,b_list)
                loss=np.mean((Yt-temp_signals)**2)
                if loss<loss_min:
                    loss_min=loss
                    parameters=temp_parameters 
            ls_parameters[i,:]=parameters
    ls_parameters[:,:2]=ls_parameters[:,:2]/1000
    return ls_parameters

def empirical_neg_log_prior(Dslow0, Dfast0, Fslow0):
    Dslow_valid = (1e-8 < np.nan_to_num(Dslow0)) & (np.nan_to_num(Dslow0) < 3 - 1e-8)
    Dfast_valid = (1e-8 < np.nan_to_num(Dfast0)) & (np.nan_to_num(Dfast0) < 100 - 1e-8)
    Fslow_valid = (1e-8 < np.nan_to_num(Fslow0)) & (np.nan_to_num(Fslow0) < 1 - 1e-8)
    valid = Dslow_valid & Dfast_valid & Fslow_valid
    Dslow0, Dfast0, Fslow0 = Dslow0[valid], Dfast0[valid], Fslow0[valid]
    
    Dslow_shape, _, Dslow_scale = stats.lognorm.fit(Dslow0, floc=0)
    Dfast_shape, _, Dfast_scale = stats.lognorm.fit(Dfast0, floc=0)
    Fslow_a, Fslow_b, _, _ = stats.beta.fit(Fslow0, floc=0, fscale=1)
    def neg_log_prior(p):
        Dslow, Dfast, Fslow, = p[0], p[1], p[2]
        if (Dfast < Dslow):
            return 1e8
        else:
            eps = 1e-8
            Dslow_prior = stats.lognorm.pdf(Dslow, Dslow_shape, scale=Dslow_scale)
            Dfast_prior = stats.lognorm.pdf(Dfast, Dfast_shape, scale=Dfast_scale)
            Fslow_prior = stats.beta.pdf(Fslow,Fslow_a, Fslow_b)
            return -np.log(Dslow_prior+eps) - np.log(Dfast_prior+eps) - np.log(Fslow_prior+eps)
    return neg_log_prior


def neg_log_likelihood(p, b, x_dw):
    return 0.5*(len(b)+1)*np.log(np.sum((equation(b, p[0], p[1], p[2])-x_dw)**2))

def neg_log_posterior(p, b, x_dw,neg_log_prior_fun):
    return neg_log_likelihood(p, b, x_dw) + neg_log_prior_fun(p)

def fit_bayesian(b_list, x_dw, x0,neg_log_prior_fun):
    try:
        if x_dw[0]<1e-8:
            return np.array([0,0,0,0]),100
        else:
            bounds = [(0., 3.), (0, 100), (0., 1.)]
            params = minimize(neg_log_posterior, x0=x0, args=(b_list, x_dw, neg_log_prior_fun), bounds=bounds)
            Dslow, Dfast, Fslow = params.x[0], params.x[1], params.x[2]
        return order(Dslow, Dfast, Fslow),params.fun
    except:
        return fit_least_squares(b_list, x_dw,x0,b_list),100

def Bayesian(signals,neg_log_prior_fun,parameters0,b_list):
    x,y=signals.shape
    bayesian_parameters=np.zeros((x, 4)) 
    for i in range(x):
        Y=signals[i,:]
        if Y[0]<1e-5:
            bayesian_parameters[i,:]=np.array([0.,0.,0.,0.])
        else:
            loss_min=1000
            for j in range(len(parameters0)): 
                p=parameters0[j]
                temp_parameters,loss=fit_bayesian(b_list,Y,p,neg_log_prior_fun)
                if loss<loss_min:
                        loss_min=loss
                        parameters=temp_parameters 
            bayesian_parameters[i,:]=parameters
    bayesian_parameters[:,:2]=bayesian_parameters[:,:2]/1000
    return bayesian_parameters

