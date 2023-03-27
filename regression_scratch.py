import warnings
warnings.filterwarnings("ignore")

import torch, random, numpy as np, pandas as pd, copy
from IPython.display import clear_output
import matplotlib.pyplot as plt

from dataset import RegressionTaskGenerator
from metalearning import MultiReptile, MultiMAML, MMAML, TSA_MAML, TaskEncoderTrainer, adapt_and_evaluate, adapt_and_evaluate_mmaml, adapt_and_evaluate_tsamaml
from models import RegressionModule, RegressionTaskEncoder, GatedNet, LSTMEmbeddingModel
from utils import DEVICE, write_in_file, evaluation_regression, plot_compare, plot_compare_sota, te_evaluation, averaging_results, n_models_comparison, plot_regression_tasks, plot_regression_results_comparison
from params_regression import *

# For reproducibility
torch.random.manual_seed(1)
np.random.seed(1)
random.seed(1)

######################## SET PARAMETERS #################################
idx_modes = [0,1,2,3,4]
all_modes_name = ["sinusoid", "linear", "quadratic", "l1norm", "tanh"]
path_dir = f"Results/regression/{k}-shot/{len(idx_modes)}_modes/hard"

steps = 100
task_mode = None
"""
################################### DATASET ######################################
print("LOAD DATASET")
tgen = RegressionTaskGenerator(k, q, modes=idx_modes) # k support points, q query points, m modes
#plot_regression_tasks(tgen, all_modes_name, n=6)

################################## TRAIN ##################################
names = ["1 model", "3 models", "5 models"]
for it in range(iters):
    clear_output(wait=True)
    print(f"Iter: {it + 1}")

    print("TRAIN SCRATCH")
    model_scratch = [RegressionModule().to(DEVICE)]

    losses_scratch = []
    id_mode = []
    for _ in range(20):
        if task_mode is None:
            nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch()
        else:
            nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch_task(task_mode)
        id_mode.append(nnewid_label)

        id_mode.append(nnewid_label)
        _, hhistory_scratch = adapt_and_evaluate(model_scratch, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, single_model=True, selection_steps=selection_steps)

        losses_scratch.append(hhistory_scratch["loss"])

    avg_losses_scratch = np.array(losses_scratch).mean(axis=0)
    write_in_file(avg_losses_scratch, f"{path_dir}/it{it+1}/avg_losses_scratch_task{task_mode}")
"""
##################################### AVERAGE RESULTS ###########################################
avg_losses = []
for it in range(iters):
    file_name = f"{path_dir}/it{it + 1}/avg_losses_scratch_task{task_mode}"
    avg_losses.append(pd.read_pickle(file_name))
avg_losses_scratch = np.mean(avg_losses, axis=0)

df_mean = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=["scratch"])
df_mean.loc["scratch"]=[round(avg_losses_scratch[1],2), round(avg_losses_scratch[10],2), round(avg_losses_scratch[99],2)]
df_mean.to_csv(f"{path_dir}/avg_comparison_scratch_task{task_mode}.csv", encoding='utf-8')
