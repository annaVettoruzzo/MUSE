import torch, copy, numpy as np
from collections import Counter
from utils import DEVICE, func_call, get_weights, interpolate, shuffle_labels
from .grade_models import grading


# -------------------------------------------------------------------
class MultiReptile():
    def __init__(self, models, loss_fn, lr, adapt_steps=1, eps=0.5, lmbd=5, eps_end=1e-5, lmbd_end=1e-5):
        self.models = models # A list of models        
        self.loss_fn = loss_fn  
        self.lr = lr
        self.adapt_steps = adapt_steps # Number of GD adaptation steps (to get task specific parameters)
        
        self.eps = eps # 0 < epsilon << 1, for interpolation
        self.eps_final = eps_end
        self.lmbd = lmbd      
        self.lmbd_final = lmbd_end # Use for weightning 
               
        self.optimizers = [torch.optim.SGD(model.parameters(), lr=lr)  for model in models]        
        # self.scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0) for optimizer in self.optimizers]
        
    # -------------------------------------------------------------------
    def fit(self, tgen, steps=10000, selection_steps=1, weightning=False, adaptive_weights=False, mode=None): #selection_steps is the number of steps used to select the best model
        best_model=[]
        for step in range(steps):
            # Sample a training task 
            if mode is None: X_sp, y_sp, X_qr, y_qr, _ = tgen.batch()
            else: X_sp, y_sp, X_qr, y_qr, _ = tgen.batch_task(mode)
            
            X = torch.cat((X_sp, X_qr))
            y = torch.cat((y_sp, y_qr))
            
            # Parameters before adaptation
            thetas = [copy.deepcopy(model.state_dict()) for model in self.models]
            
            # Grade models 
            losses = grading(self.models, self.loss_fn, self.lr, X_sp, y_sp, X_qr, y_qr, selection_steps)
            ids = np.argsort(losses)
            best_model.append(ids[0])
            
            # Compute task specific parameters with the best model
            model, opt = self.models[ids[0]], self.optimizers[ids[0]] 
            for _ in range(self.adapt_steps):
                loss = self.loss_fn(model(X), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
           
            # Parameters after adaptation (i.e. task specific parameters)
            phi = model.state_dict()
            
            # Interpolate between the meta-parameters (theta) and the task specific parameters (params)
            if weightning: # Soft version 
                for k, id in enumerate(ids):
                    weight = get_weights(k, step, steps, self.eps, self.lmbd, self.eps_final, self.lmbd_final, adaptive_weights)
                    dico = interpolate(thetas[id], phi, weight)
                    self.models[id].load_state_dict(dico)           
            else: # Hard version (i.e. winner takes all)
                id_winner = ids[0]
                dico = interpolate(thetas[id_winner], phi, self.eps)
                self.models[id_winner].load_state_dict(dico)
                
            if (step+1) % 50 == 0:
                print(f"Step: {step+1}", end="\t\r")
                
        print(f"Model selection (training): {dict(Counter(best_model))}")   
        return self