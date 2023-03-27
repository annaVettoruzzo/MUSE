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
idx_modes = [0,1,2]
all_modes_name = ["sinusoid", "linear", "quadratic", "l1norm", "tanh"]
path_dir = f"Results/regression/{k}-shot/{len(idx_modes)}_modes/hard"

################################### DATASET ######################################
print("LOAD DATASET")
tgen = RegressionTaskGenerator(k, q, modes=idx_modes) # k support points, q query points, m modes
#plot_regression_tasks(tgen, all_modes_name, n=6)

################################## TRAIN ##################################
names = ["1 model", "3 models", "5 models"]
for it in range(iters):
    clear_output(wait=True)
    print(f"Iter: {it + 1}")

    print("TRAIN REPTILE")
    n_models = 1  # Number of models that we want to create
    modelA_reptile = [RegressionModule().to(DEVICE)]  # Only one model
    #MultiReptile(modelA_reptile, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    for model in modelA_reptile: model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/model_0.pt"))

    print("TRAIN 3 MODELS REPTILE")
    n_models = 3  # Number of models
    modelsB_reptile = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    #MultiReptile(modelsB_reptile, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps*n_models, selection_steps, use_weight, adapt_weights)
    for i, model in enumerate(modelsB_reptile): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/3models_{i}.pt"))
    teB_reptile = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    #TaskEncoderTrainer(teB_reptile, modelsB_reptile, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
    teB_reptile.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/3te_model.pt"))

    print("TRAIN 5 MODELS REPTILE")
    n_models = 5  # Number of models that we want to create
    modelsC_reptile = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    #MultiReptile(modelsC_reptile, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps*n_models, selection_steps, use_weight, adapt_weights)
    for i, model in enumerate(modelsC_reptile): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/5models_{i}.pt"))
    teC_reptile = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    #TaskEncoderTrainer(teC_reptile, modelsC_reptile, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
    teC_reptile.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/5te_model.pt"))

    print("TRAIN MAML")
    modelA_maml = [RegressionModule().to(DEVICE)]  # Only one model
    #MultiMAML(modelA_maml, loss_fn, lr, adapt_steps=adapt_steps, lmbd=lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    for model in modelA_maml: model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/model_0.pt"))

    print("TRAIN 3 MODELS MAML")
    n_models = 3  # Number of models
    modelsB_maml = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    #MultiMAML(modelsB_maml, loss_fn, lr, adapt_steps=adapt_steps, lmbd=lmbd).fit(tgen, r_steps*n_models, selection_steps, use_weight, adapt_weights)
    for i, model in enumerate(modelsB_maml): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/3models_{i}.pt"))
    teB_maml = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    #TaskEncoderTrainer(teB_maml, modelsB_maml, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
    teB_maml.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/3te_model.pt"))

    print("TRAIN 5 MODELS MAML")
    n_models = 5  # Number of models that we want to create
    modelsC_maml = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    #MultiMAML(modelsC_maml, loss_fn, lr, adapt_steps=adapt_steps, lmbd=lmbd).fit(tgen, r_steps*n_models, selection_steps, use_weight, adapt_weights)
    for i, model in enumerate(modelsC_maml): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/5models_{i}.pt"))
    teC_maml = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    #TaskEncoderTrainer(teC_maml, modelsC_maml, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
    teC_maml.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/5te_model.pt"))

    print("TRAIN MMAML")
    model_mmaml = GatedNet(input_size=np.prod(1), output_size=1, hidden_sizes=[40, 40], condition_type="affine", condition_order="low2high").to(DEVICE)
    model_parameters = list(model_mmaml.parameters())
    embedding_model = LSTMEmbeddingModel(input_size=np.prod(1), output_size=1, embedding_dims=[80, 80], hidden_size=40, num_layers=2, device=DEVICE).to(DEVICE)
    embedding_parameters = list(embedding_model.parameters())
    optimizers = (torch.optim.Adam(model_parameters, lr=0.001), torch.optim.Adam(embedding_parameters, lr=0.001))
    #MMAML(model_mmaml, embedding_model, optimizers, fast_lr=lr, loss_func=loss_fn, first_order=True, num_updates=8,
    #      inner_loop_grad_clip=10, collect_accuracies=False, device=DEVICE, embedding_grad_clip=2, model_grad_clip=2).fit(tgen, r_steps)
    model_mmaml.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/model_mmaml.pt"))
    embedding_model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/embedding_model.pt"))

    print("TRAIN TSA-MAML")
    pre_model = copy.deepcopy(modelA_reptile[0])
    model_list = [RegressionModule().to(DEVICE) for _ in range(n_clusters)]
    TSA_MAML(tgen, pre_model, model_list, n_clusters, loss_fn, lr, 0.001, adapt_steps, device=DEVICE).fit(num_tasks=5000, steps=r_steps)
    #for i, model in enumerate(model_list): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/model_list_{i}_tsamaml.pt"))

    print("TRAIN MULTI-REPTILE")
    multimodel_reptile = [RegressionModule().to(DEVICE) for _ in range(len(idx_modes))]
    #for i, idx_mode in enumerate(idx_modes): MultiReptile([multimodel_reptile[i]], loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights, mode=idx_mode)
    for i, model in enumerate(multimodel_reptile): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/multi_model_{i}.pt"))

    print("TRAIN MULTI-MAML")
    multimodel_maml = [RegressionModule().to(DEVICE) for _ in range(len(idx_modes))]
    #for i, idx_mode in enumerate(idx_modes): MultiMAML([multimodel_maml[i]], loss_fn, lr, adapt_steps=adapt_steps, lmbd=lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights, mode=idx_mode)
    for i, model in enumerate(multimodel_maml): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/multi_model_{i}.pt"))

    # Save models
    torch.save(modelA_reptile[0].state_dict(), f"{path_dir}/it{it + 1}/reptile/model_{0}.pt")
    torch.save(modelA_maml[0].state_dict(), f"{path_dir}/it{it + 1}/maml/model_{0}.pt")
    for i in range(len(modelsB_reptile)): torch.save(modelsB_reptile[i].state_dict(), f"{path_dir}/it{it + 1}/reptile/3models_{i}.pt")
    torch.save(teB_reptile.state_dict(), f"{path_dir}/it{it + 1}/reptile/3te_model.pt")
    for i in range(len(modelsB_maml)): torch.save(modelsB_maml[i].state_dict(), f"{path_dir}/it{it + 1}/maml/3models_{i}.pt")
    torch.save(teB_maml.state_dict(), f"{path_dir}/it{it + 1}/maml/3te_model.pt")
    for i in range(len(modelsC_reptile)): torch.save(modelsC_reptile[i].state_dict(), f"{path_dir}/it{it + 1}/reptile/5models_{i}.pt")
    torch.save(teC_reptile.state_dict(), f"{path_dir}/it{it + 1}/reptile/5te_model.pt")
    for i in range(len(modelsC_maml)): torch.save(modelsC_maml[i].state_dict(), f"{path_dir}/it{it + 1}/maml/5models_{i}.pt")
    torch.save(teC_maml.state_dict(), f"{path_dir}/it{it + 1}/maml/5te_model.pt")
    torch.save(model_mmaml.state_dict(), f"{path_dir}/it{it + 1}/maml/model_mmaml.pt")
    torch.save(embedding_model.state_dict(), f"{path_dir}/it{it + 1}/maml/embedding_model.pt")
    for i in range(len(model_list)): torch.save(model_list[i].state_dict(), f"{path_dir}/it{it + 1}/maml/model_list_{i}_tsamaml.pt")
    for i in range(len(multimodel_reptile)): torch.save(multimodel_reptile[i].state_dict(), f"{path_dir}/it{it + 1}/reptile/multi_model_{i}.pt")
    for i in range(len(multimodel_maml)): torch.save(multimodel_maml[i].state_dict(), f"{path_dir}/it{it + 1}/maml/multi_model_{i}.pt")

    # Evaluation
    print("EVALUATION")
    # Mode 0
    (avg_losses_reptile, te_preds_reptile, te_true_reptile, avg_loracle_reptile,
     avg_losses_maml, te_preds_maml, te_true_maml, avg_loracle_maml,
     avg_losses_mmaml, avg_losses_tsamaml, avg_losses_multireptile, avg_losses_multimaml,
     eval_modes) = evaluation_regression(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile,
                                         modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml,
                                         model_mmaml, embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml,
                                         loss_fn, lr, 100, selection_steps, task_mode=0, idx_modes=idx_modes,
                                         save=f"{path_dir}/it{it + 1}")
    te_evaluation(te_preds_reptile, te_preds_maml, te_true_reptile, te_true_maml, eval_modes, idx_modes, all_modes_name, save=f"{path_dir}/it{it + 1}/TE_frequency_task0")
    # Mode 1
    (avg_losses_reptile, te_preds_reptile, te_true_reptile, avg_loracle_reptile,
     avg_losses_maml, te_preds_maml, te_true_maml, avg_loracle_maml,
     avg_losses_mmaml, avg_losses_tsamaml, avg_losses_multireptile, avg_losses_multimaml,
     eval_modes) = evaluation_regression(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile,
                                         modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml,
                                         model_mmaml, embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml,
                                         loss_fn, lr, 100, selection_steps, task_mode=1, idx_modes=idx_modes,
                                         save=f"{path_dir}/it{it + 1}")
    te_evaluation(te_preds_reptile, te_preds_maml, te_true_reptile, te_true_maml, eval_modes, idx_modes, all_modes_name, save=f"{path_dir}/it{it + 1}/TE_frequency_task1")
    # Mode 2
    (avg_losses_reptile, te_preds_reptile, te_true_reptile, avg_loracle_reptile,
     avg_losses_maml, te_preds_maml, te_true_maml, avg_loracle_maml,
     avg_losses_mmaml, avg_losses_tsamaml, avg_losses_multireptile, avg_losses_multimaml,
     eval_modes) = evaluation_regression(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile,
                                         modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml,
                                         model_mmaml, embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml,
                                         loss_fn, lr, 100, selection_steps, task_mode=2, idx_modes=idx_modes,
                                         save=f"{path_dir}/it{it + 1}")
    te_evaluation(te_preds_reptile, te_preds_maml, te_true_reptile, te_true_maml, eval_modes, idx_modes, all_modes_name, save=f"{path_dir}/it{it + 1}/TE_frequency_task2")
    # All modes
    (avg_losses_reptile, te_preds_reptile, te_true_reptile, avg_loracle_reptile,
     avg_losses_maml, te_preds_maml, te_true_maml, avg_loracle_maml,
     avg_losses_mmaml, avg_losses_tsamaml, avg_losses_multireptile, avg_losses_multimaml,
     eval_modes) = evaluation_regression(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile,
                                         modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml,
                                         model_mmaml, embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml,
                                         loss_fn, lr, 100, selection_steps, idx_modes=idx_modes,
                                         save=f"{path_dir}/it{it + 1}")
    # Plot with random tasks
    plot_compare([avg_losses_reptile, avg_loracle_reptile, avg_losses_maml, avg_loracle_maml], names, idx_modes, all_modes_name, save=f"{path_dir}/it{it + 1}/comparison.png")
    te_evaluation(te_preds_reptile, te_preds_maml, te_true_reptile, te_true_maml, eval_modes, idx_modes, all_modes_name,
                  save=f"{path_dir}/it{it + 1}/TE_frequency_taskNone")


######################################## AVERAGE RESULTS ################################
models_name_dir = ["reptile", "maml"]

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

"""
############################# DIFFERENT NUMBER OF MODELS ################################
losses_models = n_models_comparison(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile, useTE=True)
write_in_file(losses_models, f"{path_dir}/it{it+1}/reptile/losses_n_models(long)")

last_step_losses = [losses_models[j][-1] for j in range(len(losses_models))]

x = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30]
fig = plt.figure()
plt.plot(x, last_step_losses, "-.", marker="o")
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel("n_models")
plt.ylabel("Performance (on the new tasks)")
plt.title("Comparison after 100 steps")
plt.savefig(f"{path_dir}/comparison_n_models.png")


############################# FITTING CURVES ################################
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
"""
