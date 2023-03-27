import warnings

warnings.filterwarnings("ignore")

import torch, random, numpy as np, pandas as pd, copy
from IPython.display import clear_output

from dataset import ClassificationTaskGenerator
from metalearning import MultiReptile, MultiMAML, TaskEncoderTrainer, TSA_MAML, MMAML
from models import ClassificationModule, ClassificationTaskEncoder, GatedConvModel, ConvEmbeddingModel
from utils import DEVICE, evaluation_classification, plot_compare, plot_compare_sota, te_evaluation, averaging_results
from params_classification_multiple import *

# For reproducibility
torch.random.manual_seed(1)
np.random.seed(1)
random.seed(1)

################# SET PARAMETERS #####################
all_modes_name = ["FGVC_Aircraft", "FGVCx_Fungi", "CUB_Bird", "Omniglot", "CifarFS", "MiniImageNet"]
idx_modes = [3, 4, 5]

path_dir = "Results/classification_multiple/soft_adaptive"

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

    print("TRAIN REPTILE")
    modelA_reptile = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE)]  # Only one model
    MultiReptile(modelA_reptile, loss_fn, lr, adapt_steps, eps, lmbd, lmbd_end=lmbd_end).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    #for model in modelA_reptile: model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/model_0.pt"))

    print("TRAIN 3 MODELS REPTILE")
    n_models = 3
    modelsB_reptile = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE) for _ in range(n_models)]
    MultiReptile(modelsB_reptile, loss_fn, lr, adapt_steps, eps, lmbd, lmbd_end=lmbd_end).fit(tgen, r_steps*n_models, selection_steps, use_weight,adapt_weights)
    #for i, model in enumerate(modelsB_reptile): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/3models_{i}.pt"))
    teB_reptile = ClassificationTaskEncoder(k, in_dim, n_classes, n_models).to(DEVICE)
    TaskEncoderTrainer(teB_reptile, modelsB_reptile, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr, save=f"{path_dir}/TE_loss_3models.png")
    #teB_reptile.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/3te_model.pt"))

    print("TRAIN 5 MODELS REPTILE")
    n_models = 5
    modelsC_reptile = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE) for _ in range(n_models)]
    MultiReptile(modelsC_reptile, loss_fn, lr, adapt_steps, eps, lmbd, lmbd_end=lmbd_end).fit(tgen, r_steps*n_models, selection_steps, use_weight, adapt_weights)
    #for i, model in enumerate(modelsC_reptile): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/5models_{i}.pt"))
    teC_reptile = ClassificationTaskEncoder(k, in_dim, n_classes, n_models).to(DEVICE)
    TaskEncoderTrainer(teC_reptile, modelsC_reptile, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr, save=f"{path_dir}/TE_loss_5models.png")
    #teC_reptile.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/5te_model.pt"))

    print("TRAIN MAML")
    modelA_maml = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE)]  # Only one model
    MultiMAML(modelA_maml, loss_fn, lr, adapt_steps=adapt_steps, lmbd=lmbd, lmbd_end=lmbd_end).fit(tgen, r_steps)
    #for model in modelA_maml: model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/model_0.pt"))

    print("TRAIN 3 MODELS MAML")
    n_models = 3  # Number of models
    modelsB_maml = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE) for _ in range(n_models)]
    MultiMAML(modelsB_maml, loss_fn, lr, adapt_steps=adapt_steps, lmbd=lmbd, lmbd_end=lmbd_end).fit(tgen, r_steps*n_models, selection_steps, use_weight, adapt_weights)
    #for i, model in enumerate(modelsB_maml): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/3models_{i}.pt"))
    teB_maml = ClassificationTaskEncoder(k, in_dim, n_classes, n_models).to(DEVICE)
    TaskEncoderTrainer(teB_maml, modelsB_maml, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
    #teB_maml.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/3te_model.pt"))

    print("TRAIN 5 MODELS MAML")
    n_models = 5  # Number of models that we want to create
    modelsC_maml = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE) for _ in range(n_models)]
    MultiMAML(modelsC_maml, loss_fn, lr, adapt_steps=adapt_steps, lmbd=lmbd, lmbd_end=lmbd_end).fit(tgen, r_steps*n_models, selection_steps, use_weight, adapt_weights)
    #for i, model in enumerate(modelsC_maml): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/5models_{i}.pt"))
    teC_maml = ClassificationTaskEncoder(k, in_dim, n_classes, n_models).to(DEVICE)
    TaskEncoderTrainer(teC_maml, modelsC_maml, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
    #teC_maml.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/5te_model.pt"))

    print("TRAIN TSA-MAML")
    pre_model = copy.deepcopy(modelA_reptile[0])
    model_list = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE) for _ in range(n_clusters)]
    TSA_MAML(tgen, pre_model, model_list, n_clusters, loss_fn, lr, 0.001, adapt_steps, device=DEVICE).fit(num_tasks=n_tasks, steps=r_steps)
    #for i, model in enumerate(model_list): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/model_list_{i}_tsamaml.pt"))

    print("TRAIN MMAML") #as in the paper
    model_mmaml = GatedConvModel(input_channels=3, output_size=n_classes, use_max_pool=False, num_channels=32, img_side_len=84, condition_type="affine").to(DEVICE)
    model_parameters = list(model_mmaml.parameters())
    embedding_model = ConvEmbeddingModel(input_size=np.prod((3, 84, 84)), output_size=n_classes, embedding_dims=[64, 64, 64, 64], hidden_size=128,
                                         num_layers=2, convolutional=True, num_conv=4, num_channels=32, rnn_aggregation=(not True), embedding_pooling="avg", batch_norm=True,
                                         avgpool_after_conv=True, linear_before_rnn=False, img_size=(3, 84, 84)).to(DEVICE)
    embedding_parameters = list(embedding_model.parameters())
    optimizers = (torch.optim.Adam(model_parameters, lr=0.001), torch.optim.Adam(embedding_parameters, lr=0.001))
    MMAML(model_mmaml, embedding_model, optimizers, fast_lr=lr, loss_func=loss_fn, first_order=False, num_updates=5,
          inner_loop_grad_clip=20, collect_accuracies=True, device=DEVICE, embedding_grad_clip=0).fit(tgen, r_steps)
    #model_mmaml.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/model_mmaml.pt"))
    #embedding_model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/embedding_model.pt"))

    print("TRAIN MULTI-REPTILE")
    multimodel_reptile = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE) for _ in range(len(idx_modes))]
    for i, idx_mode in enumerate(idx_modes): MultiReptile([multimodel_reptile[i]], loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights, mode=idx_mode)
    #for i, model in enumerate(multimodel_reptile): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/reptile/multi_model_{i}.pt"))

    print("TRAIN MULTI-MAML")
    multimodel_maml = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE) for _ in range(len(idx_modes))]
    for i, idx_mode in enumerate(idx_modes): MultiMAML([multimodel_maml[i]], loss_fn, lr, adapt_steps=adapt_steps, lmbd=lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights, mode=idx_mode)
    #for i, model in enumerate(multimodel_maml): model.load_state_dict(torch.load(f"{path_dir}/it{it+1}/maml/multi_model_{i}.pt"))

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
    # Mode 3
    (avg_accs_reptile, te_preds_reptile, te_true_reptile, avg_oracle_reptile,
     avg_accs_maml, te_preds_maml, te_true_maml, avg_oracle_maml,
     avg_accs_mmaml, avg_accs_tsamaml, avg_accs_multireptile, avg_accs_multimaml,
     eval_modes) = evaluation_classification(tgen_eval, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile,
                                             modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml,
                                             model_mmaml, embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml,
                                             loss_fn, lr, 100, selection_steps, task_mode=3, idx_modes=idx_modes,
                                             save=f"{path_dir}/it{it + 1}")
    te_evaluation(te_preds_reptile, te_preds_maml, te_true_reptile, te_true_maml, eval_modes, idx_modes, all_modes_name, save=f"{path_dir}/it{it + 1}/TE_frequency_task3")

    # Mode 4
    (avg_accs_reptile, te_preds_reptile, te_true_reptile, avg_oracle_reptile,
     avg_accs_maml, te_preds_maml, te_true_maml, avg_oracle_maml,
     avg_accs_mmaml, avg_accs_tsamaml, avg_accs_multireptile, avg_accs_multimaml,
     eval_modes) = evaluation_classification(tgen_eval, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile,
                                             modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml,
                                             model_mmaml, embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml,
                                             loss_fn, lr, 100, selection_steps, task_mode=4, idx_modes=idx_modes,
                                             save=f"{path_dir}/it{it + 1}")
    te_evaluation(te_preds_reptile, te_preds_maml, te_true_reptile, te_true_maml, eval_modes, idx_modes, all_modes_name, save=f"{path_dir}/it{it + 1}/TE_frequency_task4")

    # Mode 5
    (avg_accs_reptile, te_preds_reptile, te_true_reptile, avg_oracle_reptile,
     avg_accs_maml, te_preds_maml, te_true_maml, avg_oracle_maml,
     avg_accs_mmaml, avg_accs_tsamaml, avg_accs_multireptile, avg_accs_multimaml,
     eval_modes) = evaluation_classification(tgen_eval, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile,
                                             modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml,
                                             model_mmaml, embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml,
                                             loss_fn, lr, 100, selection_steps, task_mode=5, idx_modes=idx_modes,
                                             save=f"{path_dir}/it{it + 1}")
    te_evaluation(te_preds_reptile, te_preds_maml, te_true_reptile, te_true_maml, eval_modes, idx_modes, all_modes_name, save=f"{path_dir}/it{it + 1}/TE_frequency_task5")

    # All modes
    (avg_accs_reptile, te_preds_reptile, te_true_reptile, avg_oracle_reptile,
     avg_accs_maml, te_preds_maml, te_true_maml, avg_oracle_maml,
     avg_accs_mmaml, avg_accs_tsamaml, avg_accs_multireptile, avg_accs_multimaml,
     eval_modes) = evaluation_classification(tgen_eval, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile,
                                             modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml,
                                             model_mmaml, embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml,
                                             loss_fn, lr, 100, selection_steps, task_mode=None, idx_modes=idx_modes,
                                             save=f"{path_dir}/it{it + 1}")
    # Compare the results
    plot_compare([avg_accs_reptile, avg_oracle_reptile, avg_accs_maml, avg_oracle_maml], names, idx_modes, all_modes_name, save=f"{path_dir}/it{it + 1}/comparison.png")
    plot_compare_sota([avg_accs_reptile, avg_oracle_reptile, avg_accs_maml, avg_oracle_maml], avg_accs_mmaml, avg_accs_tsamaml, names, idx_modes, all_modes_name, save=f"{path_dir}/it{it+1}/comparison_sota.png")
    te_evaluation(te_preds_reptile, te_preds_maml, te_true_reptile, te_true_maml, eval_modes, idx_modes, all_modes_name, save=f"{path_dir}/it{it+1}/TE_frequency")

######################################## AVERAGE RESULTS ################################
models_name_dir = ["reptile", "maml"]

model_names = ["MMAML (SOTA)", "TSA-MAML (SOTA)",
               "Reptile", "Reptile (3 models)", "Reptile (5 models)",
               "MAML", "MAML (3 models)", "MAML (5 models)",
               "Multi-Reptile", "Multi-MAML", "Reptile_3 (oracle)", "Reptile_5 (oracle)", "MAML_3 (oracle)","MAML_5 (oracle)"]

# All tasks
avg_accs, avg_accs_mmaml, avg_accs_tsamaml, avg_accs_multireptile, avg_accs_multimaml, \
std_accs, std_accs_mmaml, std_accs_tsamaml, std_accs_multireptile, std_accs_multimaml = averaging_results(path_dir, models_name_dir, task="None", metric="accs")
plot_compare_sota([avg_accs[:3], avg_accs[3:5], avg_accs[5:8], avg_accs[8:]], avg_accs_mmaml, avg_accs_tsamaml, names, idx_modes, all_modes_name, save=f"{path_dir}/avg_comparison_taskNone.png")

df_mean = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
df_std = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
avg_accs_real = [avg_accs_mmaml] + [avg_accs_tsamaml] + avg_accs[:3] + avg_accs[5:8] + [avg_accs_multireptile] + [avg_accs_multimaml] + avg_accs[3:5] + avg_accs[8:]
std_accs_real = [std_accs_mmaml] + [std_accs_tsamaml] + std_accs[:3] + std_accs[5:8] + [std_accs_multireptile] + [std_accs_multimaml] + std_accs[3:5] + std_accs[8:]

for i, name in enumerate(model_names):
    df_mean.loc[name]=[round(avg_accs_real[i][1],2), round(avg_accs_real[i][10],2), round(avg_accs_real[i][99],2)]
    df_std.loc[name]=[round(std_accs_real[i][1],2), round(std_accs_real[i][10],2), round(std_accs_real[i][99],2)]
df_mean.to_csv(f"{path_dir}/avg_comparison_taskNone.csv", encoding='utf-8')
df_std.to_csv(f"{path_dir}/std_comparison_taskNone.csv", encoding='utf-8')

# Omniglot
avg_accs, avg_accs_mmaml, avg_accs_tsamaml, avg_accs_multireptile, avg_accs_multimaml, \
std_accs, std_accs_mmaml, std_accs_tsamaml, std_accs_multireptile, std_accs_multimaml = averaging_results(path_dir, models_name_dir, task="3", metric="accs")
plot_compare_sota([avg_accs[:3], avg_accs[3:5], avg_accs[5:8], avg_accs[8:]], avg_accs_mmaml, avg_accs_tsamaml, names, idx_modes, all_modes_name, save=f"{path_dir}/avg_comparison_task3.png")

df_mean = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
df_std = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
avg_accs_real =  [avg_accs_mmaml] + [avg_accs_tsamaml] + avg_accs[:3] + avg_accs[5:8] + [avg_accs_multireptile] + [avg_accs_multimaml] + avg_accs[3:5] + avg_accs[8:]
std_accs_real = [std_accs_mmaml] + [std_accs_tsamaml] + std_accs[:3] + std_accs[5:8] + [std_accs_multireptile] + [std_accs_multimaml] + std_accs[3:5] + std_accs[8:]
for i, name in enumerate(model_names):
    df_mean.loc[name]=[round(avg_accs_real[i][1],2), round(avg_accs_real[i][10],2), round(avg_accs_real[i][99],2)]
    df_std.loc[name]=[round(std_accs_real[i][1],2), round(std_accs_real[i][10],2), round(std_accs_real[i][99],2)]
df_mean.to_csv(f"{path_dir}/avg_comparison_task3.csv", encoding='utf-8')
df_std.to_csv(f"{path_dir}/std_comparison_task3.csv", encoding='utf-8')

# CIFAR
avg_accs, avg_accs_mmaml, avg_accs_tsamaml, avg_accs_multireptile, avg_accs_multimaml, \
std_accs, std_accs_mmaml, std_accs_tsamaml, std_accs_multireptile, std_accs_multimaml = averaging_results(path_dir, models_name_dir, task="4", metric="accs")
plot_compare_sota([avg_accs[:3], avg_accs[3:5], avg_accs[5:8], avg_accs[8:]], avg_accs_mmaml, avg_accs_tsamaml, names, idx_modes, all_modes_name, save=f"{path_dir}/avg_comparison_task4.png")

df_mean = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
df_std = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
avg_accs_real =  [avg_accs_mmaml] + [avg_accs_tsamaml] + avg_accs[:3] + avg_accs[5:8] + [avg_accs_multireptile] + [avg_accs_multimaml] + avg_accs[3:5] + avg_accs[8:]
std_accs_real = [std_accs_mmaml] + [std_accs_tsamaml] + std_accs[:3] + std_accs[5:8] + [std_accs_multireptile] + [std_accs_multimaml] + std_accs[3:5] + std_accs[8:]
for i, name in enumerate(model_names):
    df_mean.loc[name]=[round(avg_accs_real[i][1],2), round(avg_accs_real[i][10],2), round(avg_accs_real[i][99],2)]
    df_std.loc[name]=[round(std_accs_real[i][1],2), round(std_accs_real[i][10],2), round(std_accs_real[i][99],2)]
df_mean.to_csv(f"{path_dir}/avg_comparison_task4.csv", encoding='utf-8')
df_std.to_csv(f"{path_dir}/std_comparison_task4.csv", encoding='utf-8')

# MiniImageNet
avg_accs, avg_accs_mmaml, avg_accs_tsamaml, avg_accs_multireptile, avg_accs_multimaml, \
std_accs, std_accs_mmaml, std_accs_tsamaml, std_accs_multireptile, std_accs_multimaml = averaging_results(path_dir, models_name_dir, task="5", metric="accs")
plot_compare_sota([avg_accs[:3], avg_accs[3:5], avg_accs[5:8], avg_accs[8:]], avg_accs_mmaml, avg_accs_tsamaml, names, idx_modes, all_modes_name, save=f"{path_dir}/avg_comparison_task5.png")

df_mean = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
df_std = pd.DataFrame(columns=["1step", "10steps", "100steps"], index=model_names)
avg_accs_real =  [avg_accs_mmaml] + [avg_accs_tsamaml] + avg_accs[:3] + avg_accs[5:8] + [avg_accs_multireptile] + [avg_accs_multimaml] + avg_accs[3:5] + avg_accs[8:]
std_accs_real = [std_accs_mmaml] + [std_accs_tsamaml] + std_accs[:3] + std_accs[5:8] + [std_accs_multireptile] + [std_accs_multimaml] + std_accs[3:5] + std_accs[8:]
for i, name in enumerate(model_names):
    df_mean.loc[name]=[round(avg_accs_real[i][1],2), round(avg_accs_real[i][10],2), round(avg_accs_real[i][99],2)]
    df_std.loc[name]=[round(std_accs_real[i][1],2), round(std_accs_real[i][10],2), round(std_accs_real[i][99],2)]
df_mean.to_csv(f"{path_dir}/avg_comparison_task5.csv", encoding='utf-8')
df_std.to_csv(f"{path_dir}/std_comparison_task5.csv", encoding='utf-8')