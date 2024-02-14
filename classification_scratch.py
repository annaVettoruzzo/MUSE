import warnings
warnings.filterwarnings("ignore")

import torch, random, numpy as np, pandas as pd, copy
from IPython.display import clear_output

from dataset import ClassificationTaskGenerator
from metalearning import MultiReptile, MultiMAML, TaskEncoderTrainer, MMAML, TSA_MAML
from models import ClassificationModule, ClassificationTaskEncoder, GatedConvModel, ConvEmbeddingModel
from utils import DEVICE, write_in_file, evaluation_classification, plot_compare, plot_compare_sota, te_evaluation, averaging_results
from params_classification_multiple import *
from metalearning import adapt_and_evaluate

# For reproducibility
torch.random.manual_seed(5)
np.random.seed(5)
random.seed(5)

################# SET PARAMETERS #####################
all_modes_name = ["FGVC_Aircraft", "FGVCx_Fungi", "CUB_Bird", "Omniglot", "CifarFS", "MiniImageNet"]
idx_modes = [3, 4, 5]

path_dir = "Results/classification_multiple/soft_adaptive"

steps = 100
task_mode = None

###################### DATASET ##########################
print("LOAD DATASET")
# To sample training tasks randomly (for meta-training)
tgen = ClassificationTaskGenerator(n_classes, k, q, idx_modes, split="train", resize=84)
# To sample new evaluation tasks (for meta-testing)
tgen_eval = ClassificationTaskGenerator(n_classes, k, q, idx_modes, split="test", resize=84)


################################## TRAIN ##################################
names = ["1 model", "3 models", "5 models"]
for it in range(iters):
    clear_output(wait=True)
    print(f"Iter: {it + 1}")

    print("TRAIN SCRATCH")
    model_scratch = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE)]

    accs_scratch = []
    id_mode = []
    for _ in range(20):
        if task_mode is None:
            nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch()
        else:
            nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch_task(task_mode)
        id_mode.append(nnewid_label)

        id_mode.append(nnewid_label)
        _, history_scratch = adapt_and_evaluate(model_scratch, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, single_model=True, acc=True, selection_steps=selection_steps)

        accs_scratch.append(history_scratch["acc"])

    avg_accs_scratch = np.array(accs_scratch).mean(axis=0)
    write_in_file(avg_accs_scratch, f"{path_dir}/it{it+1}/avg_accs_scratch_task{task_mode}")

##################################### AVERAGE RESULTS ###########################################
avg_losses = []
for it in range(iters):
    file_name = f"{path_dir}/it{it + 1}/avg_accs_scratch_task{task_mode}"
    avg_losses.append(pd.read_pickle(file_name))
avg_losses_scratch = np.mean(avg_losses, axis=0)

df_mean = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=["scratch"])
df_mean.loc["scratch"]=[round(avg_losses_scratch[1],2), round(avg_losses_scratch[10],2), round(avg_losses_scratch[99],2)]
df_mean.to_csv(f"{path_dir}/avg_comparison_scratch_task{task_mode}.csv", encoding='utf-8')
