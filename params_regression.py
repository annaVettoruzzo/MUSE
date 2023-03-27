import torch

k = 5
q = 300

iters = 4

# PARAMETERS FOR TRAINING REPTILE
loss_fn = torch.nn.MSELoss()
lr = 0.005
adapt_steps = 8 # Steps for task-specific params
r_steps = 20000 # Total training steps
selection_steps = 8 # Steps for choosing the best model (0 means adapt_grading=False)
# Weightning parameters
use_weight = False # If True use Neural Gas
adapt_weights = False # If True use Adaptive Neural Gas
eps = 0.1
lmbd = 0.2

# TASK ENCODER PARAMETERS
te_loss_fn = torch.nn.CrossEntropyLoss()
te_lr = 1e-4
te_batchsize = 35
te_steps = 1000

# PARAMETERS FOR TSA-MAML
n_clusters = 3