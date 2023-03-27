import torch

# PARAMETERS FOR THE DATA
n_classes = 5
in_dim = 3
k = 5
q = 15
iters = 3#4

loss_fn = torch.nn.CrossEntropyLoss()
lr = 0.005
adapt_steps = 5  # Steps for task-specific params
r_steps = 30000  # Total training steps

# Steps for choosing the best model (0 means adapt_grading=False)
selection_steps = 5

# Weightning parameters
use_weight = True  # If True use Neural Gas
adapt_weights = True  # If True use Adaptive Neural Gas
eps = 0.1
lmbd = 0.5
lmbd_end = 1e-5

# TASK ENCODER PARAMETERS
te_loss_fn = torch.nn.CrossEntropyLoss()
te_lr = 0.005
te_batchsize = 16
te_steps = 5000

n_clusters = 3
n_tasks = 5000
