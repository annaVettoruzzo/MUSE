import torch, copy, numpy as np, random, os
from .stateless import functional_call
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
os.system('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
gpu_number =  int(np.argmax(memory_available))
torch.cuda.set_device(gpu_number)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(gpu_number)

# -------------------------------------------------------------------
def one_hot(x, n_classes):
    x = x.cpu().numpy()
    x_hot = np.zeros((x.size, n_classes))
    x_hot[np.arange(x.size), x] = 1
    return torch.Tensor(x_hot).to(DEVICE).float()

# -------------------------------------------------------------------
def accuracy(pred, y_true):
    y_pred = pred.argmax(1).reshape(-1).cpu()
    y_true = y_true.reshape(-1).cpu()
    return accuracy_score(y_pred, y_true)

# -------------------------------------------------------------------
def interpolate(d1, d2, eps=0.1):
    return {k: d1[k] + eps * (d2[k] - d1[k]) for k in d1.keys()}

# -------------------------------------------------------------------
""" Compute models' weights inspired by Neural Gas. """
def get_weights(k, step, tot_steps, eps_start, lmbd_start, eps_final=0.005, lmbd_final=0.005, adaptive_weights=False):
    if adaptive_weights:
        epsilon_iter = eps_start * ((eps_final/eps_start)**(step/tot_steps)) 
        lmbd_iter = lmbd_start * ((lmbd_final/lmbd_start)**(step/tot_steps)) 
        weight = epsilon_iter * np.exp(-k/lmbd_iter)
    else:
        weight = eps_start * np.exp(-k/lmbd_start)  
    return weight

# -------------------------------------------------------------------
def func_call(model, params_dict, X_sp, X_qr):
    if params_dict is None: # If None, we use the params of model (so same as calling model(X) ...)
        params_dict = dict(model.named_parameters())
    X = torch.cat((X_sp, X_qr))
    y = functional_call(model, params_dict, X)
    return y[:len(X_sp)], y[len(X_sp):]

# -------------------------------------------------------------------
def shuffle_labels(X_sp, y_sp, X_qr, y_qr, n_classes):
    X, y = torch.cat((X_sp, X_qr)), torch.cat((y_sp, y_qr)).cpu().numpy()
    
    new_y = torch.tensor([0]*len(y))
    new_labels = list(range(n_classes))
    random.shuffle(new_labels)
    for i in list(range(n_classes)):
        new_y[np.where(y == i)[0]] = new_labels[i]
        
    return X[:len(X_sp)].to(DEVICE), new_y[:len(y_sp)].to(DEVICE), X[len(X_sp):].to(DEVICE), new_y[len(y_sp):].to(DEVICE)

