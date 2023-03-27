import torch, copy, numpy as np
from collections import defaultdict, OrderedDict
from utils import func_call, accuracy
from sklearn.metrics.pairwise import euclidean_distances
from .grade_models import grading



# -------------------------------------------------------------------
def adapt_and_evaluate(models, te_model, X_sp, y_sp, X_qr, y_qr, loss_fn, lr, steps=100, single_model=False, selection_steps=1, acc=False, TEprediction=True): #if TEprediction=False we select the best model regardless of the TE model   
    
    history = defaultdict(list)
    
    if not single_model: 
        # Check the real best model
        losses = grading(models, loss_fn, lr, X_sp, y_sp, X_qr, y_qr, selection_steps)
        history["te_true"] = np.argsort(losses)[0]
        # Predict the best model
        if TEprediction: 
            te_model.eval() #so we use the statistics computed before for BN
            te_output= te_model(X_sp, y_sp).detach().cpu()
            i_win = np.argmax(te_output)
        else:
            i_win = np.argsort(losses)[0]
        history["te_pred"] = i_win
        
    else: i_win=0
    
    # Copy the model (to avoid adapting the original one)
    cmodel = copy.deepcopy(models[i_win])
    
    optimizer = torch.optim.SGD(cmodel.parameters(), lr)
        
    for step in range(steps+1):
        y_sp_pred, y_qr_pred = func_call(cmodel, None, X_sp, X_qr)

        # Evaluate current model on the test data (query test) to compute the accuracy
        loss_qr = loss_fn(y_qr_pred, y_qr)
        
        history["pred"].append( y_qr_pred.cpu().detach() )
        history["loss"].append( loss_qr.cpu().detach() )
        if acc:
            acc_test = accuracy(y_qr_pred, y_qr)
            history["acc"].append(acc_test)
            
        # Adapt the model using training data (support set)
        loss = loss_fn(y_sp_pred, y_sp)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return cmodel, history


# -------------------------------------------------------------------
def adapt_and_evaluate_mmaml(model, embedding_model, X_sp, y_sp, X_qr, y_qr, loss_fn, lr=0.001, steps=100, adapt_steps=5, acc=False):
    history = defaultdict(list)

    # Copy the model (to avoid adapting the original one)
    cmodel = copy.deepcopy(model)
    
    optimizer = torch.optim.SGD(cmodel.parameters(), lr)
        
    for step in range(steps+1):
        params = OrderedDict(cmodel.named_parameters())
        embeddings = embedding_model(X_sp, y_sp)
        for i in range(adapt_steps): #num_updates
                preds_sp = cmodel(X_sp, params=params, embeddings=embeddings)
                loss_sp = loss_fn(preds_sp, y_sp)
                # Update params
                grads = torch.autograd.grad(loss_sp, params.values(), create_graph=False, allow_unused=True)
                for (name, param), grad in zip(params.items(), grads):
                    grad = grad.clamp(min=-20, max=20)
                    if grad is not None:
                        params[name] = param - 0.001 * grad
                
        y_qr_pred = cmodel(X_qr, params=params, embeddings=embeddings)
        loss_qr = loss_fn(y_qr_pred, y_qr)

        history["pred"].append(y_qr_pred.cpu().detach())
        history["loss"].append(loss_qr.cpu().detach())
        if acc:
            acc_test = accuracy(y_qr_pred, y_qr)
            history["acc"].append(acc_test)

        # Adapt the model using training data (support set)
        y_sp_pred = cmodel(X_sp, params=params, embeddings=embeddings)
        loss = loss_fn(y_sp_pred, y_sp)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return cmodel, history

# -------------------------------------------------------------------   
def sgd_step(model, params_dict, X_sp, y_sp, X_qr, loss_fn, lr_inner):
    y_sp_pred, _ = func_call(model, params_dict, X_sp, X_qr)
    inner_loss = loss_fn(y_sp_pred, y_sp)
    if params_dict == None: params_dict = dict(model.named_parameters())
    grads = torch.autograd.grad(inner_loss, params_dict.values())
    return {name: w - lr_inner * w_grad for (name, w), w_grad in zip(params_dict.items(), grads)}
    
def get_adapted_parameters(model, X, y, X_qr, loss_fn, lr_inner, adapt_steps):
    phi = sgd_step(model, None, X, y, X_qr, loss_fn, lr_inner)
    for _ in range(adapt_steps - 1):
        phi = sgd_step(model, phi, X, y, X_qr, loss_fn, lr_inner)
    return phi    

    
def adapt_and_evaluate_tsamaml(pre_model, model_list, X_sp, y_sp, X_qr, y_qr, loss_fn, lr, steps=100, adapt_steps=1, acc=False): 
    history = defaultdict(list)

    initial_weights = {}
    for i in range(len(model_list)):
        w = dict(model_list[i].named_parameters())
        initial_weights[i] = torch.cat([torch.reshape(values, [1, -1]) for _, values in w.items()], 1).cpu().detach().numpy()[0]
    initial_weights_np = np.array([weight for weight in initial_weights.values()])   
        
    # Adaptation (get the parameters adapted for this task)
    phi_tmp = get_adapted_parameters(pre_model, X_sp, y_sp, X_qr, loss_fn, lr, adapt_steps)
    phi_values = torch.cat([torch.reshape(values, [1, -1]) for _, values in phi_tmp.items()], 1).cpu().detach().numpy()

    # Compute distance
    dists = euclidean_distances(phi_values, initial_weights_np)
    model_idx = np.argmin(dists, -1)[0]
    task_specific_model = model_list[model_idx]

    # Copy the model (to avoid adapting the original one)
    cmodel = copy.deepcopy(task_specific_model)

    optimizer = torch.optim.SGD(cmodel.parameters(), lr)

    for step in range(steps):
        y_sp_pred, y_qr_pred = func_call(cmodel, None, X_sp, X_qr)

        # Evaluate current model on the test data (query test) to compute the accuracy
        loss_qr = loss_fn(y_qr_pred, y_qr)

        history["pred"].append( y_qr_pred.cpu().detach() )
        history["loss"].append( loss_qr.cpu().detach() )
        if acc:
            acc_test = accuracy(y_qr_pred, y_qr)
            history["acc"].append(acc_test)

        # Adapt the model using training data (support set)
        loss = loss_fn(y_sp_pred, y_sp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return cmodel, history
