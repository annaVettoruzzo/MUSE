import torch, numpy as np
from utils import DEVICE
import matplotlib.pyplot as plt
from .grade_models import grading

# -------------------------------------------------------------------
class TaskEncoderTrainer:
    def __init__(self, te, models, loss_fn, te_loss_fn, te_lr=0.001):
        self.te = te # The task encoder model
        self.models = models 
        self.loss_fn = loss_fn
        self.te_loss_fn = te_loss_fn
        self.te_optimizer = torch.optim.Adam(self.te.parameters(), lr=te_lr)        
    
    # -------------------------------------------------------------------
    def create_batch(self, tgen, batch_size, selection_steps, lr):
        datasets, y_models = [], []
        
        Xs_sp, ys_sp = [], [] 
        for _ in range(batch_size):
            X_sp, y_sp, X_qr, y_qr, _ = tgen.batch()       
            
            # Assign model label to the task 
            losses = grading(self.models, self.loss_fn, lr, X_sp, y_sp, X_qr, y_qr, selection_steps)
            i_win = np.argmin(losses)
            
            y_models.append(i_win)
            Xs_sp.extend(X_sp)
            ys_sp.extend(y_sp)
            
        y_models = torch.Tensor(y_models).long().to(DEVICE)
        return torch.stack(Xs_sp), torch.stack(ys_sp), y_models

    # -------------------------------------------------------------------
    def encode(self, x, y):
        return self.te(x, y)
    
    # -------------------------------------------------------------------
    def fit(self, tgen, batch_size=10, steps=10000, selection_steps=1, lr=0.005, save=""):
        #selection_steps is the number of steps used to select the best model
        losses = []
        for step in range(steps):
            Xs_sp, ys_sp, y_models = self.create_batch(tgen, batch_size, selection_steps, lr)
            
            y_pred = self.encode(Xs_sp, ys_sp)
            loss = self.te_loss_fn(y_pred, y_models)
            
            self.te_optimizer.zero_grad()
            loss.backward()
            self.te_optimizer.step()
            
            losses.append(loss.item())
            
            if (step+1) % 100 == 0:
                print(f"Step: {step+1},   Loss: {loss.item()}", end="\t\r")

        if save:
            plt.figure()
            plt.plot(losses)
            plt.title("Task Encoder loss")
            plt.savefig(save)

        return self
    
    
