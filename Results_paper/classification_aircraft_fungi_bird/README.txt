MUSE-R is run with:

# PARAMETERS
n_classes = 5
in_dim = 3
k = 5
q = 20
iters = 3
loss_fn = torch.nn.CrossEntropyLoss()
lr = 0.005
adapt_steps = 5
r_steps = 40000
selection_steps = 5
use_weight = True
adapt_weights = True 
eps = 0.1
lmbd = 0.5
lmbd_end = 1e-5

MUSE-M is run with:

# PARAMETERS
n_classes = 5
in_dim = 3
k = 5
q = 20
iters = 3
loss_fn = torch.nn.CrossEntropyLoss()
lr = 0.005
adapt_steps = 7 # Steps for task-specific params
r_steps = 40000  # Total training steps
# Steps for choosing the best model (0 means adapt_grading=False)
selection_steps = 7
# Weightning parameters
use_weight = False  # If True use Neural Gas
adapt_weights = False  # If True use Adaptive Neural Gas
eps = 0.1
lmbd = 0.5
lmbd_end = 1e-5