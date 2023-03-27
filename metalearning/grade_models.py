import torch, copy
import torch.nn.functional as F
from utils import func_call

# -------------------------------------------------------------------
def grade_models_std(models, loss_fn, X_sp, y_sp, X_qr, y_qr):
    X = torch.cat((X_sp, X_qr))
    y = torch.cat((y_sp, y_qr))

    losses = torch.stack([ loss_fn(model(X), y) for model in models ])
    return losses.detach().cpu().numpy()

# -------------------------------------------------------------------
def model_adaptation(model, loss_fn, lr, X_sp, y_sp, X_qr, y_qr, selection_steps):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for step in range(selection_steps):
        y_pred_sp, _ = func_call(model, None, X_sp, X_qr)
        loss = loss_fn(y_pred_sp, y_sp)
        opt.zero_grad()
        loss.backward()
        opt.step()         
        
# -------------------------------------------------------------------
def grade_models_adapt(models, loss_fn, lr, X_sp, y_sp, X_qr, y_qr, selection_steps):
    losses = []
    for model in models:
        cmodel = copy.deepcopy(model) # to avoid modifying the original model
        model_adaptation(cmodel, loss_fn, lr, X_sp, y_sp, X_qr, y_qr, selection_steps) # adapt the parameters towards a specific task (with support set)
        _, y_pred_qr = func_call(cmodel, None, X_sp, X_qr) # evaluate the model (with query set)
        loss = loss_fn(y_pred_qr, y_qr).detach()
        losses.append(loss)
    losses = torch.stack(losses)
    return losses.detach().cpu().numpy()

# -------------------------------------------------------------------
def grading(models, loss_fn, lr, X_sp, y_sp, X_qr, y_qr, selection_steps=1):
    if selection_steps>0: 
        return grade_models_adapt(models, loss_fn, lr, X_sp, y_sp, X_qr, y_qr, selection_steps)
    else:
        return grade_models_std(models, loss_fn, X_sp, y_sp, X_qr, y_qr)
        