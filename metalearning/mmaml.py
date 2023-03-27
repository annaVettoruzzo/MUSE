from collections import OrderedDict
import torch, copy, numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm


# -------------------------------------------------------------------
class MMAML:
    def __init__(self, model, embedding_model, optimizers, fast_lr, loss_func,
                 first_order, num_updates, inner_loop_grad_clip,
                 collect_accuracies, device, embedding_grad_clip=0,
                 model_grad_clip=0):
        self._model = model
        self._embedding_model = embedding_model
        self._fast_lr = fast_lr
        self._optimizers = optimizers
        self._loss_func = loss_func
        self._first_order = first_order
        self._num_updates = num_updates
        self._inner_loop_grad_clip = inner_loop_grad_clip
        self._collect_accuracies = collect_accuracies
        self._device = device
        self._embedding_grad_clip = embedding_grad_clip
        self._model_grad_clip = model_grad_clip
        self._grads_mean = []
        self.tbs = 1

    # -------------------------------------------------------------------
    def update_params(self, loss, params):
        """Apply one step of gradient descent on the loss function `loss`,
        with step-size `self._fast_lr`, and returns the updated parameters.
        """
        create_graph = not self._first_order
        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=create_graph, allow_unused=True)
        for (name, param), grad in zip(params.items(), grads):
            if self._inner_loop_grad_clip > 0 and grad is not None:
                grad = grad.clamp(min=-self._inner_loop_grad_clip,
                                  max=self._inner_loop_grad_clip)
            if grad is not None:
                params[name] = param - self._fast_lr * grad
        return params
    
    # -------------------------------------------------------------------
    def adapt(self, tgen):
        adapted_params = []
        embeddings_list = []
        tot_loss=[]
        for i in range(self.tbs):
            # Get a random task: sample support set and query set
            X_sp, y_sp, X_qr, y_qr, _ = tgen.batch()
            
            params = OrderedDict(self._model.named_parameters())
            embeddings = self._embedding_model(X_sp, y_sp, None)
            for i in range(self._num_updates):
                preds_sp = self._model(X_sp, params=params, embeddings=embeddings)
                loss_sp = self._loss_func(preds_sp, y_sp)
                params = self.update_params(loss_sp, params=params)
              
            preds_qr = self._model(X_qr, params=params, embeddings=embeddings)
            loss_qr = self._loss_func(preds_qr, y_qr)
            
            tot_loss.append(loss_qr)
            adapted_params.append(params)
            embeddings_list.append(embeddings)
         
        mean_loss = torch.mean(torch.stack(tot_loss))
        for optimizer in self._optimizers:
            optimizer.zero_grad()
        mean_loss.backward()

        self._optimizers[0].step()

        if len(self._optimizers) > 1:
            if self._embedding_grad_clip > 0:
                _grad_norm = clip_grad_norm_(self._embedding_model.parameters(), self._embedding_grad_clip)
            else:
                _grad_norm = get_grad_norm(self._embedding_model.parameters())
            # grad_norm
            self._grads_mean.append(_grad_norm)
            self._optimizers[1].step()
            
        return mean_loss, adapted_params, embeddings_list        
    
    
    # -------------------------------------------------------------------
    """
    Train MMAML.
    """
    def fit(self, tgen, steps=10000):
        tot_loss=0
        for step in range(steps):
            loss, adapted_params, embeddings = self.adapt(tgen)
            
            if (step+1)%50 == 0:
                print(f"Step: {step+1}, loss: {loss:.5f}", end="\t\r")
        return