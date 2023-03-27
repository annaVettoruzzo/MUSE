import torch, copy, numpy as np
from collections import Counter
from utils import DEVICE, func_call, get_weights, shuffle_labels
from .grade_models import grading

# -------------------------------------------------------------------
class MultiMAML():
    def __init__(self, models, loss_fn, lr_inner, lr_outer=0.001, adapt_steps=1, lmbd=1, lmbd_end=1e-5):
        self.models = models # A list of models        
        self.loss_fn = loss_fn  
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.adapt_steps = adapt_steps # Number of GD adaptation steps (to get task specific parameters)
        
        self.lmbd = lmbd
        self.lmbd_final = lmbd_end 
        
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=lr_outer)  for model in models]        
        
        
    # -------------------------------------------------------------------
    def set_learning_rate(self, optimizer, weight):
        for g in optimizer.param_groups: 
            g['lr'] = self.lr_outer * weight
        return
    
    # -------------------------------------------------------------------
    def adapt(self, model, params_dict, X_sp, y_sp, X_qr):
        y_sp_pred, _ = func_call(model, params_dict, X_sp, X_qr)
        inner_loss = self.loss_fn(y_sp_pred, y_sp)
        
        if params_dict == None: params_dict = dict(model.named_parameters())
        grads = torch.autograd.grad(inner_loss, params_dict.values())
        return {name: w - self.lr_inner * w_grad for (name, w), w_grad in zip(params_dict.items(), grads)}
    
    # -------------------------------------------------------------------
    def get_adapted_parameters(self, model, X_sp, y_sp, X_qr):
        phi = self.adapt(model, None, X_sp, y_sp, X_qr)
        for _ in range(self.adapt_steps - 1):
            phi = self.adapt(model, phi,X_sp, y_sp, X_qr)
        return phi

    # -------------------------------------------------------------------
    def train_maml(self, model, optimizer, X_sp, y_sp, X_qr, y_qr):
        phi = self.get_adapted_parameters(model, X_sp, y_sp, X_qr)  # Adaptation (get the parameters adapted for this task)
        _, y_qr_pred = func_call(model, phi, X_sp, X_qr)
        loss = self.loss_fn(y_qr_pred, y_qr)  # Loss of phi on the query set
        
        # Optimize tot_loss with respect to theta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return 
    
    # -------------------------------------------------------------------
    def fit(self, tgen, steps=10000, selection_steps=1, weightning=False, adaptive_weights=False, mode=None): #selection_steps is the number of steps used to select the best model
        best_model = []
        for step in range(steps):
            
            # Sample a training task 
            if mode is None: X_sp, y_sp, X_qr, y_qr, _ = tgen.batch()
            else: X_sp, y_sp, X_qr, y_qr, _ = tgen.batch_task(mode)
            
            #if self.classification_simple: X_sp, y_sp, X_qr, y_qr = shuffle_labels(X_sp, y_sp, X_qr, y_qr, self.n_classes)

            # Grade models 
            losses = grading(self.models, self.loss_fn, self.lr_inner, X_sp, y_sp, X_qr, y_qr, selection_steps)
            ids = np.argsort(losses)
            best_model.append(ids[0])
            
            # Compute task specific params with the best model
            if weightning:
                for k, id in enumerate(ids):
                    model, opt = self.models[id], self.optimizers[id]
                    # Compute weight and change learning rate
                    weight = get_weights(k, step, steps, eps_start=1, lmbd_start=self.lmbd, lmbd_final=self.lmbd_final, adaptive_weights=adaptive_weights)
                    self.set_learning_rate(opt, weight)
                    # Train model with MAML
                    self.train_maml(model, opt, X_sp, y_sp, X_qr, y_qr)       
            else:
                model, opt = self.models[ids[0]], self.optimizers[ids[0]] 
                self.train_maml(model, opt, X_sp, y_sp, X_qr, y_qr)
                
            if (step+1) % 50 == 0:
                print(f"Step: {step+1}", end="\t\r")
        print(f"Model selection (training): {dict(Counter(best_model))}")   
        return