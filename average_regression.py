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
torch.random.manual_seed(5)
np.random.seed(5)
random.seed(5)

######################## SET PARAMETERS #################################
idx_modes = [0,1,2]
k=10
it = 2
all_modes_name = ["sinusoid", "linear", "quadratic", "l1norm", "tanh"]
path_dir = f"Results/regression/{k}-shot/{len(idx_modes)}_modes/hard/it{it}"
print(path_dir)
######################################## AVERAGE RESULTS ################################
names = ["1 model", "3 models", "5 models"]
models_name_dir = ["reptile", "maml"]

"""
def averaging_results(path_dir, models_name_dir, task="None", metric="losses"): #metric is "losses" for regr and "accs" for class
    # Average losses and std for MReptile and MMAML
    avg_losses = []
    std_losses = []
    for name in models_name_dir:
        losses, losses_oracle = [], []
        for i in [0, 2, 3]:
            file_name = f"{path_dir}/it{i + 1}/{name}/avg_{metric}_task{task}"
            losses.append(pd.read_pickle(file_name))
            file_name = f"{path_dir}/it{i + 1}/{name}/avg_{metric}_oracle_task{task}"
            losses_oracle.append(pd.read_pickle(file_name))
        avg_losses.extend(np.mean(losses, axis=0))
        avg_losses.extend(np.mean(losses_oracle, axis=0))
        std_losses.extend(np.std(losses, axis=0))
        std_losses.extend(np.std(losses_oracle, axis=0))

    # Average losses and std for SOTA and oracles
    losses_mmaml, losses_tsamaml, losses_multireptile, losses_multimaml = [], [], [], []
    for i in [0, 2, 3]:
        file_name = f"{path_dir}/it{i + 1}/maml/avg_{metric}_mmaml_task{task}"
        losses_mmaml.append(pd.read_pickle(file_name))
        file_name = f"{path_dir}/it{i + 1}/maml/avg_{metric}_tsamaml_task{task}"
        losses_tsamaml.append(pd.read_pickle(file_name))
        file_name = f"{path_dir}/it{i + 1}/reptile/avg_{metric}_multireptile_task{task}"
        losses_multireptile.append(pd.read_pickle(file_name))
        file_name = f"{path_dir}/it{i + 1}/maml/avg_{metric}_multimaml_task{task}"
        losses_multimaml.append(pd.read_pickle(file_name))
    avg_losses_mmaml = np.mean(losses_mmaml, axis=0)
    avg_losses_tsamaml = np.mean(losses_tsamaml, axis=0)
    avg_losses_multireptile = np.mean(losses_multireptile, axis=0)
    avg_losses_multimaml = np.mean(losses_multimaml, axis=0)
    std_losses_mmaml = np.std(losses_mmaml, axis=0)
    std_losses_tsamaml = np.std(losses_tsamaml, axis=0)
    std_losses_multireptile = np.std(losses_multireptile, axis=0)
    std_losses_multimaml = np.std(losses_multimaml, axis=0)
    return avg_losses, avg_losses_mmaml, avg_losses_tsamaml, avg_losses_multireptile, avg_losses_multimaml, \
           std_losses, std_losses_mmaml, std_losses_tsamaml, std_losses_multireptile, std_losses_multimaml

model_names = ["MMAML (SOTA)", "TSA-MAML (SOTA)",
               "Reptile", "Reptile (3 models)", "Reptile (5 models)",
               "MAML", "MAML (3 models)", "MAML (5 models)",
               "Multi-Reptile", "Multi-MAML", "Reptile_3 (oracle)", "Reptile_5 (oracle)", "MAML_3 (oracle)","MAML_5 (oracle)"]

# All tasks
avg_losses, avg_losses_mmaml, avg_losses_tsamaml, avg_losses_multireptile, avg_losses_multimaml, \
std_losses, std_losses_mmaml, std_losses_tsamaml, std_losses_multireptile, std_losses_multimaml = averaging_results(path_dir, models_name_dir, task="None")
plot_compare_sota([avg_losses[:3], avg_losses[3:5], avg_losses[5:8], avg_losses[8:]], avg_losses_mmaml, avg_losses_tsamaml, names, idx_modes, all_modes_name, save=f"{path_dir}/avg_comparison_taskNone.png")

df_mean = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
df_std = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
avg_losses_real =  [avg_losses_mmaml] + [avg_losses_tsamaml] + avg_losses[:3] + avg_losses[5:8] + [avg_losses_multireptile] + [avg_losses_multimaml] + avg_losses[3:5] + avg_losses[8:]
std_losses_real = [std_losses_mmaml] + [std_losses_tsamaml] + std_losses[:3] + std_losses[5:8] + [std_losses_multireptile] + [std_losses_multimaml] + std_losses[3:5] + std_losses[8:]
for i, name in enumerate(model_names):
    df_mean.loc[name]=[round(avg_losses_real[i][1],2), round(avg_losses_real[i][10],2), round(avg_losses_real[i][99],2)]
    df_std.loc[name]=[round(std_losses_real[i][1],2), round(std_losses_real[i][10],2), round(std_losses_real[i][99],2)]
df_mean.to_csv(f"{path_dir}/avg_comparison_taskNone.csv", encoding='utf-8')
df_std.to_csv(f"{path_dir}/std_comparison_taskNone.csv", encoding='utf-8')

# Sinusoidal
avg_losses, avg_losses_mmaml, avg_losses_tsamaml, avg_losses_multireptile, avg_losses_multimaml, \
           std_losses, std_losses_mmaml, std_losses_tsamaml, std_losses_multireptile, std_losses_multimaml = averaging_results(path_dir, models_name_dir, task="0")
plot_compare_sota([avg_losses[:3], avg_losses[3:5], avg_losses[5:8], avg_losses[8:]], avg_losses_mmaml, avg_losses_tsamaml, names, idx_modes, all_modes_name, save=f"{path_dir}/avg_comparison_task0.png")

df_mean = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
df_std = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
avg_losses_real =  [avg_losses_mmaml] + [avg_losses_tsamaml] + avg_losses[:3] + avg_losses[5:8] + [avg_losses_multireptile] + [avg_losses_multimaml] + avg_losses[3:5] + avg_losses[8:]
std_losses_real = [std_losses_mmaml] + [std_losses_tsamaml] + std_losses[:3] + std_losses[5:8] + [std_losses_multireptile] + [std_losses_multimaml] + std_losses[3:5] + std_losses[8:]
for i, name in enumerate(model_names):
    df_mean.loc[name]=[round(avg_losses_real[i][1],2), round(avg_losses_real[i][10],2), round(avg_losses_real[i][99],2)]
    df_std.loc[name]=[round(std_losses_real[i][1],2), round(std_losses_real[i][10],2), round(std_losses_real[i][99],2)]
df_mean.to_csv(f"{path_dir}/avg_comparison_task0.csv", encoding='utf-8')
df_std.to_csv(f"{path_dir}/std_comparison_task0.csv", encoding='utf-8')

# Linear
avg_losses, avg_losses_mmaml, avg_losses_tsamaml, avg_losses_multireptile, avg_losses_multimaml, \
           std_losses, std_losses_mmaml, std_losses_tsamaml, std_losses_multireptile, std_losses_multimaml = averaging_results(path_dir, models_name_dir, task="1")
plot_compare_sota([avg_losses[:3], avg_losses[3:5], avg_losses[5:8], avg_losses[8:]], avg_losses_mmaml, avg_losses_tsamaml, names, idx_modes, all_modes_name, save=f"{path_dir}/avg_comparison_task1.png")

df_mean = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
df_std = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
avg_losses_real =  [avg_losses_mmaml] + [avg_losses_tsamaml] + avg_losses[:3] + avg_losses[5:8] + [avg_losses_multireptile] + [avg_losses_multimaml] + avg_losses[3:5] + avg_losses[8:]
std_losses_real = [std_losses_mmaml] + [std_losses_tsamaml] + std_losses[:3] + std_losses[5:8] + [std_losses_multireptile] + [std_losses_multimaml] + std_losses[3:5] + std_losses[8:]
for i, name in enumerate(model_names):
    df_mean.loc[name]=[round(avg_losses_real[i][1],2), round(avg_losses_real[i][10],2), round(avg_losses_real[i][99],2)]
    df_std.loc[name]=[round(std_losses_real[i][1],2), round(std_losses_real[i][10],2), round(std_losses_real[i][99],2)]
df_mean.to_csv(f"{path_dir}/avg_comparison_task1.csv", encoding='utf-8')
df_std.to_csv(f"{path_dir}/std_comparison_task1.csv", encoding='utf-8')

# Quadratic
avg_losses, avg_losses_mmaml, avg_losses_tsamaml, avg_losses_multireptile, avg_losses_multimaml, \
           std_losses, std_losses_mmaml, std_losses_tsamaml, std_losses_multireptile, std_losses_multimaml = averaging_results(path_dir, models_name_dir, task="1")
plot_compare_sota([avg_losses[:3], avg_losses[3:5], avg_losses[5:8], avg_losses[8:]], avg_losses_mmaml, avg_losses_tsamaml, names, idx_modes, all_modes_name, save=f"{path_dir}/avg_comparison_task2.png")

df_mean = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
df_std = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
avg_losses_real =  [avg_losses_mmaml] + [avg_losses_tsamaml] + avg_losses[:3] + avg_losses[5:8] + [avg_losses_multireptile] + [avg_losses_multimaml] + avg_losses[3:5] + avg_losses[8:]
std_losses_real = [std_losses_mmaml] + [std_losses_tsamaml] + std_losses[:3] + std_losses[5:8] + [std_losses_multireptile] + [std_losses_multimaml] + std_losses[3:5] + std_losses[8:]
for i, name in enumerate(model_names):
    df_mean.loc[name]=[round(avg_losses_real[i][1],2), round(avg_losses_real[i][10],2), round(avg_losses_real[i][99],2)]
    df_std.loc[name]=[round(std_losses_real[i][1],2), round(std_losses_real[i][10],2), round(std_losses_real[i][99],2)]
df_mean.to_csv(f"{path_dir}/avg_comparison_task2.csv", encoding='utf-8')
df_std.to_csv(f"{path_dir}/std_comparison_task2.csv", encoding='utf-8')


############################# FITTING CURVES ################################
# Load dataset
print("LOAD DATASET")
tgen = RegressionTaskGenerator(k, q, modes=idx_modes) # k support points, q query points, m modes

modelA_reptile = [RegressionModule().to(DEVICE)] # Only one model
for model in modelA_reptile: model.load_state_dict(torch.load(f"{path_dir}/reptile/model_0.pt"))

modelA_maml = [RegressionModule().to(DEVICE)]  # Only one model
for model in modelA_maml: model.load_state_dict(torch.load(f"{path_dir}/maml/model_0.pt"))

modelsB_reptile = [RegressionModule().to(DEVICE) for _ in range(3)]
for i, model in enumerate(modelsB_reptile): model.load_state_dict(torch.load(f"{path_dir}/reptile/3models_{i}.pt"))
teB_reptile = RegressionTaskEncoder(k=k, out_dim=3).to(DEVICE)
teB_reptile.load_state_dict(torch.load(f"{path_dir}/reptile/3te_model.pt"))

modelsC_reptile = [RegressionModule().to(DEVICE) for _ in range(5)]
for i, model in enumerate(modelsC_reptile): model.load_state_dict(torch.load(f"{path_dir}/reptile/5models_{i}.pt"))
teC_reptile = RegressionTaskEncoder(k=k, out_dim=5).to(DEVICE)
teC_reptile.load_state_dict(torch.load(f"{path_dir}/reptile/5te_model.pt"))

modelsB_maml = [RegressionModule().to(DEVICE) for _ in range(3)]
for i, model in enumerate(modelsB_maml): model.load_state_dict(torch.load(f"{path_dir}/maml/3models_{i}.pt"))
teB_maml = RegressionTaskEncoder(k=k, out_dim=3).to(DEVICE)
teB_maml.load_state_dict(torch.load(f"{path_dir}/maml/3te_model.pt"))

modelsC_maml = [RegressionModule().to(DEVICE) for _ in range(5)]
for i, model in enumerate(modelsC_maml): model.load_state_dict(torch.load(f"{path_dir}/maml/5models_{i}.pt"))
teC_maml = RegressionTaskEncoder(k=k, out_dim=5).to(DEVICE)
teC_maml.load_state_dict(torch.load(f"{path_dir}/maml/5te_model.pt"))

newX_sp, newy_sp, newX_qr, newy_qr, newid_label = tgen.batch()  # New task used for evaluation
# Compare the results with 5 models, Reptile and MAML
teC_reptile.eval()
predC_reptile = teC_reptile(newX_sp, newy_sp).detach().cpu().numpy()
best_modelC_reptile = np.argmax(predC_reptile)
teC_maml.eval()
predC_maml = teC_maml(newX_sp, newy_sp).detach().cpu().numpy()
best_modelC_maml = np.argmax(predC_maml)

_, history_bestmodelC_reptile = adapt_and_evaluate([modelsC_reptile[best_modelC_reptile]], None, newX_sp, newy_sp, newX_qr, newy_qr, loss_fn, lr, steps=100, single_model=True, selection_steps=selection_steps)
_, history_reptile = adapt_and_evaluate(modelA_reptile, None, newX_sp, newy_sp, newX_qr, newy_qr, loss_fn, lr, steps=100, single_model=True, selection_steps=selection_steps)
_, history_bestmodelC_maml = adapt_and_evaluate([modelsC_maml[best_modelC_maml]], None, newX_sp, newy_sp, newX_qr, newy_qr, loss_fn, lr, steps=100, single_model=True, selection_steps=selection_steps)
_, history_maml = adapt_and_evaluate(modelA_maml, None, newX_sp, newy_sp, newX_qr, newy_qr, loss_fn, lr, steps=100, single_model=True, selection_steps=selection_steps)

histories = [history_bestmodelC_reptile, history_reptile, history_bestmodelC_maml, history_maml]
plot_regression_results_comparison(newX_sp, newy_sp, newX_qr, newy_qr, histories, best_modelC_reptile, predC_reptile[0][best_modelC_reptile], best_modelC_maml, predC_maml[0][best_modelC_maml], save=f"{path_dir}/fitting_curves.png")

############################# DIFFERENT NUMBER OF MODELS ################################
print("N_MODELS")
losses_models = n_models_comparison(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile, save=f"{path_dir}/n_models", useTE=True)
write_in_file(losses_models, f"{path_dir}/losses_n_models_TEtrue")

last_step_losses = [losses_models[j][-1] for j in range(len(losses_models))]

x = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30]
fig = plt.figure()
plt.plot(x, last_step_losses, "-.", marker="o")
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel("n_models")
plt.ylabel("Performance (on the new tasks)")
plt.title("Comparison after 100 steps")
plt.savefig(f"{path_dir}/comparison_n_models_TEtrue.png")

############################# DIFFERENT NUMBER OF MODELS w/0 TE ################################
print("N_MODELS")
losses_models = n_models_comparison(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile, save=f"{path_dir}/n_models", useTE=False)
write_in_file(losses_models, f"{path_dir}/reptile/losses_n_models_TEfalse")

last_step_losses = [losses_models[j][-1] for j in range(len(losses_models))]

x = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30]
fig = plt.figure()
plt.plot(x, last_step_losses, "-.", marker="o")
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel("n_models")
plt.ylabel("Performance (on the new tasks)")
plt.title("Comparison after 100 steps")
plt.savefig(f"{path_dir}/comparison_n_models_TEfalse.png")
"""

x = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20]
file_name = f"{path_dir}/n_models/losses_n_models_TEtrue"
losses_n_models = pd.read_pickle(file_name)
last_step_losses = [losses_n_models[j][-1] for j in range(len(losses_n_models)-2)]

fig = plt.figure()
plt.plot(x, last_step_losses, "-.", marker="o")
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel("$\it{N}$")
plt.ylabel("MSE on the new tasks")
#plt.title("Comparison after 100 steps")
plt.savefig(f"{path_dir}/n_models/comparison_n_models_TEtrue.png")