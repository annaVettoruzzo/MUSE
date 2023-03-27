import numpy as np, pandas as pd, pickle, seaborn as sns, matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from sklearn.metrics import accuracy_score
from metalearning import MultiReptile, TaskEncoderTrainer, adapt_and_evaluate, adapt_and_evaluate_mmaml, \
    adapt_and_evaluate_tsamaml
from models import RegressionModule, RegressionTaskEncoder
from utils import DEVICE
from params_regression import * #FIXME


# -------------------------------------------------------------------
def write_in_file(file, file_directory):
    a_file = open(file_directory, "wb+")
    pickle.dump(file, a_file)
    a_file.close()

# -------------------------------------------------------------------
def plot_regression_tasks(tgen, all_modes_name, n=16):
    for i in range(n):
        X_sp, y_sp, X_qr, y_qr, label = tgen.batch()
        plt.scatter(X_sp.cpu(), y_sp.cpu())
        plt.plot(X_qr.cpu(), y_qr.cpu(), label=all_modes_name[label])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


# EVALUATION MODELS-------------------------------------------------------------------
# -------------------------------------------------------------------
def evaluation_regression(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile,
                          modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml, model_mmaml,
                          embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml,
                          loss_fn, lr, steps=100, selection_steps=1, task_mode=None, idx_modes=[0, 1, 2, 3, 4],
                          save=""):
    losses_reptile, losses_maml, loracle_reptile, loracle_maml, losses_mmaml, losses_tsamaml, losses_multireptile, losses_multimaml = [], [], [], [], [], [], [], []
    te_preds_reptile, te_preds_maml = [], []
    te_true_reptile, te_true_maml = [], []
    id_mode = []
    for _ in range(20):
        # Get a new test task
        if task_mode is None:
            nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch()
        else:
            nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch_task(task_mode)
        id_mode.append(nnewid_label)

        # Adapt reptile
        _, hhistoryA_reptile = adapt_and_evaluate(modelA_reptile, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, single_model=True, selection_steps=selection_steps)
        _, hhistoryB_reptile = adapt_and_evaluate(modelsB_reptile, teB_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps)
        _, hhistoryC_reptile = adapt_and_evaluate(modelsC_reptile, teC_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps)
        losses_reptile.append([hhistoryA_reptile["loss"], hhistoryB_reptile["loss"], hhistoryC_reptile["loss"]])
        te_preds_reptile.append([hhistoryB_reptile["te_pred"], hhistoryC_reptile["te_pred"]])
        te_true_reptile.append([hhistoryB_reptile["te_true"], hhistoryC_reptile["te_true"]])

        # Adapt reptile (oracle = without using task encoder) 
        _, horacleB_reptile = adapt_and_evaluate(modelsB_reptile, teB_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, TEprediction=False)
        _, horacleC_reptile = adapt_and_evaluate(modelsC_reptile, teC_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, TEprediction=False)
        loracle_reptile.append([horacleB_reptile["loss"], horacleC_reptile["loss"]])

        # Adapt maml
        _, hhistoryA_maml = adapt_and_evaluate(modelA_maml, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, single_model=True, selection_steps=selection_steps)
        _, hhistoryB_maml = adapt_and_evaluate(modelsB_maml, teB_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps)
        _, hhistoryC_maml = adapt_and_evaluate(modelsC_maml, teC_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps)
        losses_maml.append([hhistoryA_maml["loss"], hhistoryB_maml["loss"], hhistoryC_maml["loss"]])
        te_preds_maml.append([hhistoryB_maml["te_pred"], hhistoryC_maml["te_pred"]])
        te_true_maml.append([hhistoryB_maml["te_true"], hhistoryC_maml["te_true"]])

        # Adapt maml (oracle = without using task encoder) 
        _, horacleB_maml = adapt_and_evaluate(modelsB_maml, teB_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, TEprediction=False)
        _, horacleC_maml = adapt_and_evaluate(modelsC_maml, teC_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, TEprediction=False)
        loracle_maml.append([horacleB_maml["loss"], horacleC_maml["loss"]])

        # Adapt mmaml (sota)
        _, hhistory_mmaml = adapt_and_evaluate_mmaml(model_mmaml, embedding_model, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, adapt_steps=selection_steps, acc=False)
        losses_mmaml.append(hhistory_mmaml["loss"])

        # Adapt tsa-maml (sota)
        _, hhistory_tsamaml = adapt_and_evaluate_tsamaml(pre_model, model_list, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, adapt_steps=selection_steps, acc=False)
        losses_tsamaml.append(hhistory_tsamaml["loss"])

        # Adapt multi-reptile and multi-maml (oracle)
        idx = idx_modes.index(nnewid_label)
        _, hhistory_multireptile = adapt_and_evaluate([multimodel_reptile[idx]], _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, single_model=True, selection_steps=selection_steps)
        _, hhistory_multimaml = adapt_and_evaluate([multimodel_maml[idx]], _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, single_model=True, selection_steps=selection_steps)
        losses_multireptile.append(hhistory_multireptile["loss"])
        losses_multimaml.append(hhistory_multimaml["loss"])

    avg_losses_reptile = np.array(losses_reptile).mean(axis=0)
    avg_losses_maml = np.array(losses_maml).mean(axis=0)

    avg_loracle_reptile = np.array(loracle_reptile).mean(axis=0)
    avg_loracle_maml = np.array(loracle_maml).mean(axis=0)

    avg_losses_mmaml = np.array(losses_mmaml).mean(axis=0)
    avg_losses_tsamaml = np.array(losses_tsamaml).mean(axis=0)

    avg_losses_multireptile = np.array(losses_multireptile).mean(axis=0)
    avg_losses_multimaml = np.array(losses_multimaml).mean(axis=0)

    if save:
        write_in_file(avg_losses_reptile, f"{save}/reptile/avg_losses_task{task_mode}")
        write_in_file(avg_loracle_reptile, f"{save}/reptile/avg_losses_oracle_task{task_mode}")
        write_in_file(avg_losses_maml, f"{save}/maml/avg_losses_task{task_mode}")
        write_in_file(avg_loracle_maml, f"{save}/maml/avg_losses_oracle_task{task_mode}")
        write_in_file(avg_losses_mmaml, f"{save}/maml/avg_losses_mmaml_task{task_mode}")
        write_in_file(avg_losses_tsamaml, f"{save}/maml/avg_losses_tsamaml_task{task_mode}")
        write_in_file(te_preds_reptile, f"{save}/reptile/te_preds_task{task_mode}")
        write_in_file(te_preds_maml, f"{save}/maml/te_preds_task{task_mode}")
        write_in_file(te_true_reptile, f"{save}/reptile/te_true_task{task_mode}")
        write_in_file(te_true_maml, f"{save}/maml/te_true_task{task_mode}")
        write_in_file(avg_losses_multireptile, f"{save}/reptile/avg_losses_multireptile_task{task_mode}")
        write_in_file(avg_losses_multimaml, f"{save}/maml/avg_losses_multimaml_task{task_mode}")

    return (avg_losses_reptile, te_preds_reptile, te_true_reptile, avg_loracle_reptile,
            avg_losses_maml, te_preds_maml, te_true_maml, avg_loracle_maml,
            avg_losses_mmaml, avg_losses_tsamaml, avg_losses_multireptile, avg_losses_multimaml, id_mode)


# -------------------------------------------------------------------
def evaluation_classification(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile,
                                             modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml,
                                             model_mmaml, embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml,
                                             loss_fn, lr, steps=100, selection_steps=1, task_mode=None, idx_modes=[0, 1, 2, 3, 4], save=""):
    accs_reptile, accs_maml, oracle_reptile, oracle_maml, accs_mmaml, accs_tsamaml, accs_multireptile, accs_multimaml = [], [], [], [], [], [], [], []
    te_preds_reptile, te_preds_maml = [], []
    te_true_reptile, te_true_maml = [], []
    id_mode = []
    for _ in range(20):
        # Get a new test task
        if task_mode is None:
            nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch()
        else:
            nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch_task(task_mode)
        id_mode.append(nnewid_label)

        # Adapt reptile
        _, hhistoryA_reptile = adapt_and_evaluate(modelA_reptile, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn,
                                                  lr, steps, single_model=True, acc=True, selection_steps=selection_steps)
        _, hhistoryB_reptile = adapt_and_evaluate(modelsB_reptile, teB_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr,
                                                  loss_fn, lr, steps, selection_steps=selection_steps, acc=True)
        _, hhistoryC_reptile = adapt_and_evaluate(modelsC_reptile, teC_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr,
                                                  loss_fn, lr, steps, selection_steps=selection_steps, acc=True)
        accs_reptile.append([hhistoryA_reptile["acc"], hhistoryB_reptile["acc"], hhistoryC_reptile["acc"]])
        te_preds_reptile.append([hhistoryB_reptile["te_pred"], hhistoryC_reptile["te_pred"]])
        te_true_reptile.append([hhistoryB_reptile["te_true"], hhistoryC_reptile["te_true"]])

        # Adapt reptile (oracle = without using task encoder) 
        _, horacleB_reptile = adapt_and_evaluate(modelsB_reptile, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn,
                                                 lr, steps, selection_steps=selection_steps, acc=True, TEprediction=False)
        _, horacleC_reptile = adapt_and_evaluate(modelsC_reptile, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn,
                                                 lr, steps, selection_steps=selection_steps, acc=True, TEprediction=False)
        oracle_reptile.append([horacleB_reptile["acc"], horacleC_reptile["acc"]])

        # Adapt maml
        _, hhistoryA_maml = adapt_and_evaluate(modelA_maml, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr,
                                               steps, single_model=True, selection_steps=selection_steps, acc=True)
        _, hhistoryB_maml = adapt_and_evaluate(modelsB_maml, teB_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn,
                                               lr, steps, selection_steps=selection_steps, acc=True)
        _, hhistoryC_maml = adapt_and_evaluate(modelsC_maml, teC_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn,
                                               lr, steps, selection_steps=selection_steps, acc=True)
        accs_maml.append([hhistoryA_maml["acc"], hhistoryB_maml["acc"], hhistoryC_maml["acc"]])
        te_preds_maml.append([hhistoryB_maml["te_pred"], hhistoryC_maml["te_pred"]])
        te_true_maml.append([hhistoryB_maml["te_true"], hhistoryC_maml["te_true"]])

        # Adapt maml (oracle = without using task encoder) 
        _, horacleB_maml = adapt_and_evaluate(modelsB_maml, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr,
                                              steps, selection_steps=selection_steps, acc=True, TEprediction=False)
        _, horacleC_maml = adapt_and_evaluate(modelsC_maml, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr,
                                              steps, selection_steps=selection_steps, acc=True, TEprediction=False)
        oracle_maml.append([horacleB_maml["acc"], horacleC_maml["acc"]])

        # Adapt mmaml (sota)
        _, hhistory_mmaml = adapt_and_evaluate_mmaml(model_mmaml, embedding_model, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr,
                                                     loss_fn, lr, steps, adapt_steps=selection_steps, acc=True)
        accs_mmaml.append(hhistory_mmaml["acc"])

        # Adapt tsa-maml (sota)
        _, hhistory_tsamaml = adapt_and_evaluate_tsamaml(pre_model, model_list, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr,
                                                         loss_fn, lr, steps, adapt_steps=selection_steps, acc=True)
        accs_tsamaml.append(hhistory_tsamaml["acc"])

        # Adapt multi-reptile and multi-maml (oracle)
        idx = idx_modes.index(nnewid_label)
        _, hhistory_multireptile = adapt_and_evaluate([multimodel_reptile[idx]], _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr,
                                                      loss_fn, lr, steps, single_model=True, selection_steps=selection_steps, acc=True)
        _, hhistory_multimaml = adapt_and_evaluate([multimodel_maml[idx]], _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr,
                                                   loss_fn, lr, steps, single_model=True, selection_steps=selection_steps, acc=True)
        accs_multireptile.append(hhistory_multireptile["acc"])
        accs_multimaml.append(hhistory_multimaml["acc"])

    avg_accs_reptile = np.array(accs_reptile).mean(axis=0)
    avg_accs_maml = np.array(accs_maml).mean(axis=0)
    avg_oracle_reptile = np.array(oracle_reptile).mean(axis=0)
    avg_oracle_maml = np.array(oracle_maml).mean(axis=0)
    avg_accs_mmaml = np.array(accs_mmaml).mean(axis=0)
    avg_accs_tsamaml = np.array(accs_tsamaml).mean(axis=0)
    avg_accs_multireptile = np.array(accs_multireptile).mean(axis=0)
    avg_accs_multimaml = np.array(accs_multimaml).mean(axis=0)

    if save:
        write_in_file(avg_accs_reptile, f"{save}/reptile/avg_accs_task{task_mode}")
        write_in_file(avg_oracle_reptile, f"{save}/reptile/avg_accs_oracle_task{task_mode}")
        write_in_file(avg_accs_maml, f"{save}/maml/avg_accs_task{task_mode}")
        write_in_file(avg_oracle_maml, f"{save}/maml/avg_accs_oracle_task{task_mode}")
        write_in_file(avg_accs_mmaml, f"{save}/maml/avg_accs_mmaml_task{task_mode}")
        write_in_file(avg_accs_tsamaml, f"{save}/maml/avg_accs_tsamaml_task{task_mode}")
        write_in_file(te_preds_reptile, f"{save}/reptile/te_preds_task{task_mode}")
        write_in_file(te_preds_maml, f"{save}/maml/te_preds_task{task_mode}")
        write_in_file(te_true_reptile, f"{save}/reptile/te_true_task{task_mode}")
        write_in_file(te_true_maml, f"{save}/maml/te_true_task{task_mode}")
        write_in_file(avg_accs_multireptile, f"{save}/reptile/avg_accs_multireptile_task{task_mode}")
        write_in_file(avg_accs_multimaml, f"{save}/maml/avg_accs_multimaml_task{task_mode}")

    return (avg_accs_reptile, te_preds_reptile, te_true_reptile, avg_oracle_reptile,
            avg_accs_maml, te_preds_maml, te_true_maml, avg_oracle_maml,
            avg_accs_mmaml, avg_accs_tsamaml, avg_accs_multireptile, avg_accs_multimaml, id_mode)


# EVALUATION TASK ENCODER -------------------------------------------------------------------
# -------------------------------------------------------------------
def te_evaluation(te_preds_reptile, te_preds_maml, te_true_reptile, te_true_maml, eval_modes, idx_modes, all_modes_name, save=""):
    teB_preds_reptile = [item[0] for item in te_preds_reptile]
    teC_preds_reptile = [item[1] for item in te_preds_reptile]
    teB_true_reptile = [item[0] for item in te_true_reptile]
    teC_true_reptile = [item[1] for item in te_true_reptile]

    teB_preds_maml = [item[0] for item in te_preds_maml]
    teC_preds_maml = [item[1] for item in te_preds_maml]
    teB_true_maml = [item[0] for item in te_true_maml]
    teC_true_maml = [item[1] for item in te_true_maml]

    teB_accuracy_reptile = accuracy_score(teB_preds_reptile, teB_true_reptile)
    teB_accuracy_maml = accuracy_score(teB_preds_maml, teB_true_maml)

    teC_accuracy_reptile = accuracy_score(teC_preds_reptile, teC_true_reptile)
    teC_accuracy_maml = accuracy_score(teC_preds_maml, teC_true_maml)

    df = pd.DataFrame(columns=["TE_MReptile", "TE_MMAML"], index=["3 models", "5 models"])
    df.loc["3 models"] = [teB_accuracy_reptile, teB_accuracy_maml]
    df.loc["5 models"] = [teC_accuracy_reptile, teC_accuracy_maml]

    frequency_te(teB_preds_reptile, teB_true_reptile, eval_modes, idx_modes, all_modes_name, n_models=3, method="MReptile", save=f"{save}_MReptile_3.jpg")
    frequency_te(teC_preds_reptile, teC_true_reptile, eval_modes, idx_modes, all_modes_name, n_models=5, method="MReptile", save=f"{save}_MReptile_5.jpg")

    frequency_te(teB_preds_maml, teB_true_maml, eval_modes, idx_modes, all_modes_name, n_models=3, method="MMAML", save=f"{save}_MMAML_3.jpg")
    frequency_te(teC_preds_maml, teC_true_maml, eval_modes, idx_modes, all_modes_name, n_models=5, method="MMAML", save=f"{save}_MMAML_5.jpg")
    return df


# -------------------------------------------------------------------   
def frequency_te(te_preds, te_true, mode_labels, idx_modes, all_modes_name, n_models, method="", save=""):  # use jpg to save
    # Compute accuracy
    te_acc = accuracy_score(te_preds, te_true)

    dictionary_pred = {}
    dictionary_true = {}
    for i in idx_modes:
        idxs = list(np.where(np.array(mode_labels) == i)[0])

        models_pred = np.array(te_preds)[idxs]
        frequency_pred = dict(Counter(models_pred))

        models_true = np.array(te_true)[idxs]
        frequency_true = dict(Counter(models_true))

        for k in range(n_models):
            frequency_pred.setdefault(k, 0)
            frequency_true.setdefault(k, 0)

        frequency_pred = OrderedDict(sorted(frequency_pred.items()))
        frequency_true = OrderedDict(sorted(frequency_true.items()))

        dictionary_pred[i] = frequency_pred
        dictionary_true[i] = frequency_true

    cols_name = [all_modes_name[i] for i in idx_modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    sns.heatmap(pd.DataFrame.from_dict(dictionary_pred), xticklabels=cols_name, cmap="Blues", annot=True, cbar=False,
                ax=ax1)
    ax1.title.set_size(9)
    ax1.title.set_text('Prediction')
    sns.heatmap(pd.DataFrame.from_dict(dictionary_true), xticklabels=cols_name, cmap="Blues", annot=True, cbar=False,
                ax=ax2)
    ax2.title.set_size(9)
    ax2.title.set_text('Groundtruth')
    plt.suptitle(f"{method} Task Encoder Frequency - accuracy = {te_acc}", fontsize=11)
    if save: plt.savefig(save)
    plt.show()


# MODEL COMPARISON -------------------------------------------------------------------
# -------------------------------------------------------------------
def n_models_comparison(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile, save="", useTE=True):
    # 2 models
    n_models = 2
    models2 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models2, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model2 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 4 models
    n_models = 4
    models4 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models4, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model4 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 6 models
    n_models = 6
    models6 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models6, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model6 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 7 models
    n_models = 7
    models7 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models7, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model7 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 8 models
    n_models = 8
    models8 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models8, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model8 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 10 models
    n_models = 10
    models10 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models10, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model10 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 12 models
    n_models = 12
    models12 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models12, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model12 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 15 models
    n_models = 15
    models15 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models15, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model15 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 18 models
    n_models = 18
    models18 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models18, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model18 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 20 models
    n_models = 20
    models20 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models20, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model20 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 25 models
    n_models = 25
    models25 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models25, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model25 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)
    # 30 models
    n_models = 30
    models30 = [RegressionModule().to(DEVICE) for _ in range(n_models)]
    MultiReptile(models30, loss_fn, lr, adapt_steps, eps, lmbd).fit(tgen, r_steps, selection_steps, use_weight, adapt_weights)
    te_model30 = RegressionTaskEncoder(k=k, out_dim=n_models).to(DEVICE)

    if save:
        for i in range(len(models2)): torch.save(models2[i].state_dict(), f"{save}/models2_{i}.pt")
        for i in range(len(models4)): torch.save(models4[i].state_dict(), f"{save}/models4_{i}.pt")
        for i in range(len(models6)): torch.save(models6[i].state_dict(), f"{save}/models6_{i}.pt")
        for i in range(len(models7)): torch.save(models7[i].state_dict(), f"{save}/models7_{i}.pt")
        for i in range(len(models8)): torch.save(models8[i].state_dict(), f"{save}/models8_{i}.pt")
        for i in range(len(models10)): torch.save(models10[i].state_dict(), f"{save}/models10_{i}.pt")
        for i in range(len(models12)): torch.save(models12[i].state_dict(), f"{save}/models12_{i}.pt")
        for i in range(len(models15)): torch.save(models15[i].state_dict(), f"{save}/models15_{i}.pt")
        for i in range(len(models18)): torch.save(models18[i].state_dict(), f"{save}/models18_{i}.pt")
        for i in range(len(models20)): torch.save(models20[i].state_dict(), f"{save}/models20_{i}.pt")
        for i in range(len(models25)): torch.save(models25[i].state_dict(), f"{save}/models25_{i}.pt")
        for i in range(len(models30)): torch.save(models30[i].state_dict(), f"{save}/models30_{i}.pt")

    if useTE:
        TaskEncoderTrainer(te_model2, models2, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model4, models4, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model6, models6, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model7, models7, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model8, models8, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model10, models10, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model12, models12, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model15, models15, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model18, models18, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model20, models20, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model25, models25, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)
        TaskEncoderTrainer(te_model30, models30, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps, selection_steps, lr)

    (loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss10, loss12, loss15, loss18, loss20, loss25, loss30) = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(20):
        # Get a new test task
        nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch()

        # Adapt the models A, B, C, D, to the new task
        _, hhistory1 = adapt_and_evaluate(modelA_reptile, 0, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr,
                                          steps=100, single_model=True, selection_steps=selection_steps,
                                          TEprediction=useTE)
        _, hhistory2 = adapt_and_evaluate(models2, te_model2, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory3 = adapt_and_evaluate(modelsB_reptile, teB_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory4 = adapt_and_evaluate(models4, te_model4, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory5 = adapt_and_evaluate(modelsC_reptile, teC_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory6 = adapt_and_evaluate(models6, te_model6, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory7 = adapt_and_evaluate(models7, te_model7, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory8 = adapt_and_evaluate(models8, te_model8, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory10 = adapt_and_evaluate(models10, te_model10, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory12 = adapt_and_evaluate(models12, te_model12, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory15 = adapt_and_evaluate(models15, te_model15, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory18 = adapt_and_evaluate(models18, te_model18, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory20 = adapt_and_evaluate(models20, te_model20, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory25 = adapt_and_evaluate(models25, te_model25, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)
        _, hhistory30 = adapt_and_evaluate(models30, te_model30, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps=100, selection_steps=selection_steps, TEprediction=useTE)

        loss1.append(hhistory1["loss"])
        loss2.append(hhistory2["loss"])
        loss3.append(hhistory3["loss"])
        loss4.append(hhistory4["loss"])
        loss5.append(hhistory5["loss"])
        loss6.append(hhistory6["loss"])
        loss7.append(hhistory7["loss"])
        loss8.append(hhistory8["loss"])
        loss10.append(hhistory10["loss"])
        loss12.append(hhistory12["loss"])
        loss15.append(hhistory15["loss"])
        loss18.append(hhistory18["loss"])
        loss20.append(hhistory20["loss"])
        loss25.append(hhistory25["loss"])
        loss30.append(hhistory30["loss"])

    return [np.array(loss1).mean(axis=0), np.array(loss2).mean(axis=0), np.array(loss3).mean(axis=0), np.array(loss4).mean(axis=0), np.array(loss5).mean(axis=0), np.array(loss6).mean(axis=0),
            np.array(loss7).mean(axis=0), np.array(loss8).mean(axis=0), np.array(loss10).mean(axis=0), np.array(loss12).mean(axis=0), np.array(loss15).mean(axis=0), np.array(loss18).mean(axis=0),
            np.array(loss20).mean(axis=0), np.array(loss25).mean(axis=0), np.array(loss30).mean(axis=0)]


# AVERAGE RESULTS-------------------------------------------------------------------
# -------------------------------------------------------------------
def averaging_results(path_dir, models_name_dir, task="None", metric="losses"): #metric is "losses" for regr and "accs" for class
    # Average losses and std for MReptile and MMAML
    avg_losses = []
    std_losses = []
    for name in models_name_dir:
        losses, losses_oracle = [], []
        for i in range(iters):
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
    for i in range(iters):
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


# PLOTS-------------------------------------------------------------------
# -------------------------------------------------------------------
def plot_compare_sota(losses, losses_mmaml, losses_tsamaml, names, idx_modes, all_modes_name, save=""):  # use jpg to save
    title_plot = ""
    for i in idx_modes: title_plot = title_plot + all_modes_name[i] + " "

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    for arr, name in zip(losses[0], names):
        ax1.plot(arr[1:], "-", linewidth=1.5, label=name)
    for arr, name in zip(losses[1], names[1:]):
        ax1.plot(arr[1:], "--", label=f"{name} (oracle)")
    ax1.plot(losses_mmaml, label="MMAML (SOTA)")
    ax1.plot(losses_tsamaml, label="TSA-MAML (SOTA)")
    ax1.title.set_text("Reptile")
    ax1.set_ylabel("Performance (on the new tasks)")
    ax1.set_xlabel("Adaptation steps (on the new tasks)")
    ax1.legend()
    for arr, name in zip(losses[2], names):
        ax2.plot(arr[1:], "-", linewidth=1.5, label=name)
    for arr, name in zip(losses[3], names[1:]):
        ax2.plot(arr[1:], "--", label=f"{name} (oracle)")
    ax2.plot(losses_mmaml, label="MMAML (SOTA)")
    ax2.plot(losses_tsamaml, label="TSA-MAML (SOTA)")
    ax2.title.set_text("MAML")
    ax2.legend()
    ax2.set_ylabel("Performance (on the new tasks)")
    ax2.set_xlabel("Adaptation steps (on the new tasks)")
    plt.ylim(bottom=0)
    plt.suptitle(title_plot, fontsize=12)
    if save: plt.savefig(save)
    plt.show()


# -------------------------------------------------------------------
def plot_compare(losses, names, idx_modes, all_modes_name, save=""):  # use jpg to save
    title_plot = ""
    for i in idx_modes: title_plot = title_plot + all_modes_name[i] + " "

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    for arr, name in zip(losses[0], names):
        ax1.plot(arr[1:], "-", linewidth=1.5, label=name)
    for arr, name in zip(losses[1], names[1:]):
        ax1.plot(arr[1:], "--", label=f"{name} (oracle)")
    ax1.title.set_text("Reptile")
    ax1.set_ylabel("Performance (on the new tasks)")
    ax1.set_xlabel("Adaptation steps (on the new tasks)")
    ax1.legend()
    for arr, name in zip(losses[2], names):
        ax2.plot(arr[1:], "-", linewidth=1.5, label=name)
    for arr, name in zip(losses[3], names[1:]):
        ax2.plot(arr[1:], "--", label=f"{name} (oracle)")
    ax2.title.set_text("MAML")
    ax2.legend()
    ax2.set_ylabel("Performance (on the new tasks)")
    ax2.set_xlabel("Adaptation steps (on the new tasks)")
    plt.ylim(bottom=0)
    plt.suptitle(title_plot, fontsize=12)
    if save: plt.savefig(save)
    plt.show()


# -------------------------------------------------------------------
def plot_model_knowledge(X, y, X_test, y_test, history_preds,
                         save=""):  # history_preds=history["pred"] for all the models
    fig, ax = plt.subplots(1, len(history_preds), figsize=(20, 4))
    X, y, X_test, y_test = [arr.cpu().detach() for arr in [X, y, X_test, y_test]]
    for i in range(len(history_preds)):
        ax[i].plot(X_test, history_preds[i][0], '--')
        ax[i].set_title(f"model_{i}")
    fig.suptitle("Meta-learned functions - after 0 adaptation steps")
    if save: plt.savefig(save)
    plt.show()


# -------------------------------------------------------------------
def plot_regression_results(X, y, X_test, y_test, history):
    X, y, X_test, y_test = [arr.cpu().detach() for arr in [X, y, X_test, y_test]]
    losses, preds = history["loss"], history["pred"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax1.plot(losses, marker='.')
    ax2.plot(X_test, y_test, label="True Function")
    ax2.plot(X, y, '^', c="r", label="Training Points")
    ax2.plot(X_test, preds[0], '--', label="After 0 steps")
    ax2.plot(X_test, preds[10], '--', label="After 10 steps")
    ax2.plot(X_test, preds[-1], '--', label=f"After {len(preds) - 1} steps")
    ax1.set_xlabel("Adaptation steps (for new task)")
    ax1.set_ylabel("Test loss (for new task)")
    ax2.legend()
    plt.show()


# -------------------------------------------------------------------
def plot_regression_results_comparison(X, y, X_test, y_test, histories, bestmodel_idx_reptile, score_idx_reptile, bestmodel_idx_maml, score_idx_maml, save=""):
    X, y, X_test, y_test = [arr.cpu().detach() for arr in [X, y, X_test, y_test]]
    preds = [h["pred"] for h in histories]

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.tight_layout(pad=5.0)
    axs = axs.ravel()
    for i in range(len(histories)):
        axs[i].plot(X_test, y_test, label="True Function")
        axs[i].plot(X, y, '^', c="r", label="Training Points")
        axs[i].plot(X_test, preds[i][0], '--', label="After 0 steps")
        axs[i].plot(X_test, preds[i][10], '--', label="After 10 steps")
        axs[i].plot(X_test, preds[i][-1], '--', label=f"After {len(preds[0]) - 1} steps")
    axs[0].title.set_text(
        f"MReptile - model {bestmodel_idx_reptile} \n Prediction probability = {round(score_idx_reptile * 100, 2)}%")
    axs[1].title.set_text('Reptile')
    axs[2].title.set_text(
        f"MMAML - model {bestmodel_idx_maml} \n Prediction probability = {round(score_idx_maml * 100, 2)}%")
    axs[3].title.set_text('MAML')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save: plt.savefig(save, bbox_inches='tight')
    plt.show()
