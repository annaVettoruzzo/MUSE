import math, torch, numpy as np
from utils.train_utils import func_call, DEVICE
from utils.train_utils import func_call
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

class TSA_MAML:
    def __init__(self, tgen, pre_model, model_list, n_clusters, loss_fn, lr_inner, lr_outer=0.001, adapt_steps=1, task_batch_size=1, device=DEVICE):
        self.tgen = tgen
        self.model = pre_model #fixed 
        self.model_list = model_list #optimized
        self.loss_fn = loss_fn
        self.lr_inner = lr_inner
        self.adapt_steps = adapt_steps  # Number of GD adaptation steps (to get task specific parameters)
        self.tbs = task_batch_size
        self.n_clusters = n_clusters
        
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=lr_outer) for model in self.model_list]  # To optimize the meta-parameters theta
        
        self.device = device
    # -------------------------------------------------------------------
    """
    Takes a dictionary of named params and a dataset (X, y), and 
    returns the updated named params after taking one gradient descent step
    """
    def sgd_step(self, model, params_dict, X_sp, y_sp, X_qr):
        y_sp_pred, _ = func_call(model, params_dict, X_sp, X_qr)
        inner_loss = self.loss_fn(y_sp_pred, y_sp)
        
        if params_dict == None: params_dict = dict(model.named_parameters())
        grads = torch.autograd.grad(inner_loss, params_dict.values())
        return {name: w - self.lr_inner * w_grad for (name, w), w_grad in zip(params_dict.items(), grads)}
    
    # -------------------------------------------------------------------
    """
    Takes a support set (X, y) corresponding to a specific task, and returns the task specific 
    parameters phi (after adapting theta with GD using one or multiple adaptation steps)
    """
    def get_adapted_parameters(self, model, X, y, X_qr):
        phi = self.sgd_step(model, None, X, y, X_qr)
        for _ in range(self.adapt_steps - 1):
            phi = self.sgd_step(model, phi, X, y, X_qr)
        return phi    
    
    # -------------------------------------------------------------------
    """
    Initiialize models with centroids
    """
    def initialize_models(self, initial_weights):
        shape_w = [pre_v.cpu().numpy().shape for _, pre_v in self.model.state_dict().items()]
        length_w = [math.prod(shapes) for shapes in shape_w]
        for model_idx, model_w in initial_weights.items():
            start_i = 0
            j = 0
            for k, v in self.model_list[model_idx].named_parameters():
                w = model_w[start_i:start_i+length_w[j]]
                w = np.reshape(w, shape_w[j])
                v.data = torch.nn.parameter.Parameter(torch.tensor(w)).to(self.device)
                start_i += length_w[j]
                j += 1
        return self 
    
    # -------------------------------------------------------------------
    def get_initial_weights(self, num_tasks):
        solutions = []
        for _ in range(num_tasks):
            phi_values = []
            
            # Get a random task: sample support set and query set
            X_sp, y_sp, X_qr, y_qr, _ = self.tgen.batch()

            phi = self.get_adapted_parameters(self.model, X_sp, y_sp, X_qr)  # Adaptation (get the parameters adapted for this task)

            phi_values = torch.cat([torch.reshape(values, [1, -1]) for _, values in phi.items()], 1)
            solutions.append(phi_values.cpu().detach())

        # clustering solutions
        S = np.vstack(solutions)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(S.squeeze())
        solutions_labels = kmeans.labels_
        
        # compute centroids
        initial_weights = {}
        for i in range(self.n_clusters):
            initial_weights[i] = S[solutions_labels==i].mean(0)
        
        # initialize models
        self.initialize_models(initial_weights)
        
        return initial_weights
        
    # -------------------------------------------------------------------
    def fit(self, num_tasks=10000, steps=10000):
        # Initialize models
        initial_weights = self.get_initial_weights(num_tasks)
        initial_weights_np = np.array([weight for weight in initial_weights.values()])

        for step in range(steps):
            tot_loss = [0 for _ in range(self.n_clusters)]  # Will contain the average loss for a mini-batch of tasks
            num_tasks = [0 for _ in range(self.n_clusters)] 
            for i in range(self.tbs):
                # Get a random task: sample support set and query set
                X_sp, y_sp, X_qr, y_qr, _ = self.tgen.batch()
                
                # Fine-tune theta* 
                phi_tmp = self.get_adapted_parameters(self.model, X_sp, y_sp, X_qr)  
                phi_values = torch.cat([torch.reshape(values, [1, -1]) for _, values in phi_tmp.items()], 1).cpu().detach().numpy()

                # Compute distance and select the cluster (i.e. task-specific model)
                dists = euclidean_distances(phi_values, initial_weights_np)
                model_idx = np.argmin(dists, -1)[0]
                task_specific_model = self.model_list[model_idx]

                # Fine-tune theta_id (actual inner loop)
                phi = self.get_adapted_parameters(task_specific_model, X_sp, y_sp, X_qr)  
                
                _, y_qr_pred = func_call(task_specific_model, phi, X_sp, X_qr)
                loss = self.loss_fn(y_qr_pred, y_qr)  # Loss of phi on the query set
                
                # Sum the loss over the batch of tasks
                tot_loss[model_idx] += loss
                num_tasks[model_idx] += 1

            #tot_loss = [i / j for i, j in zip(tot_loss, num_tasks)] # Average loss (doesn't really matter)
            
            # Optimize tot_loss with respect to theta
            for i, optimizer in enumerate(self.optimizers):
                if tot_loss[i] == 0: continue
                
                optimizer.zero_grad()
                tot_loss[i].backward()
                optimizer.step()
