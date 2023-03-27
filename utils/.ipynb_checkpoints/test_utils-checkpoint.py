import numpy as np, pandas as pd, pickle
from collections import Counter, OrderedDict
import seaborn as sns
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from metalearning import adapt_and_evaluate, adapt_and_evaluate_mmaml, adapt_and_evaluate_tsamaml

# -------------------------------------------------------------------
def write_in_file(file, file_directory):
    a_file = open(file_directory, "wb")
    pickle.dump(file, a_file)
    a_file.close()
               
# -------------------------------------------------------------------
def evaluation_regression(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile, \
                          modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml, model_mmaml, \
                          embedding_model, pre_model, model_list, multimodel_reptile, multimodel_maml, \
                          loss_fn, lr, steps=100, selection_steps=1, task_mode=None, idx_modes=[0,1,2,3,4], save=""):
    losses_reptile, losses_maml, loracle_reptile, loracle_maml, losses_mmaml, losses_tsamaml, losses_multireptile, losses_multimaml = [], [], [], [], [], [], [], []
    te_preds_reptile, te_preds_maml = [], []
    te_true_reptile, te_true_maml = [], []
    id_mode = []
    for _ in range(20):
        # Get a new test task
        if task_mode is None: nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch()
        else: nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch_task(task_mode)
        id_mode.append(nnewid_label)
        
        # Adapt reptile
        _, hhistoryA_reptile = adapt_and_evaluate(modelA_reptile, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, single_model=True, selection_steps=selection_steps)
        _, hhistoryB_reptile = adapt_and_evaluate(modelsB_reptile, teB_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps)
        _, hhistoryC_reptile = adapt_and_evaluate(modelsC_reptile, teC_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps)
        losses_reptile.append( [hhistoryA_reptile["loss"], hhistoryB_reptile["loss"], hhistoryC_reptile["loss"]] )
        te_preds_reptile.append( [hhistoryB_reptile["te_pred"], hhistoryC_reptile["te_pred"]] )
        te_true_reptile.append( [hhistoryB_reptile["te_true"], hhistoryC_reptile["te_true"]] )

        # Adapt reptile (oracle = without using task encoder) 
        _, horacleB_reptile = adapt_and_evaluate(modelsB_reptile, teB_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, TEprediction=False)
        _, horacleC_reptile = adapt_and_evaluate(modelsC_reptile, teC_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, TEprediction=False)
        loracle_reptile.append([horacleB_reptile["loss"], horacleC_reptile["loss"]] )
        
        # Adapt maml
        _, hhistoryA_maml = adapt_and_evaluate(modelA_maml, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, single_model=True, selection_steps=selection_steps)
        _, hhistoryB_maml = adapt_and_evaluate(modelsB_maml, teB_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps)
        _, hhistoryC_maml = adapt_and_evaluate(modelsC_maml, teC_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps)
        losses_maml.append( [hhistoryA_maml["loss"], hhistoryB_maml["loss"], hhistoryC_maml["loss"]] )
        te_preds_maml.append( [hhistoryB_maml["te_pred"], hhistoryC_maml["te_pred"]] )
        te_true_maml.append( [hhistoryB_maml["te_true"], hhistoryC_maml["te_true"]] )

        # Adapt maml (oracle = without using task encoder) 
        _, horacleB_maml = adapt_and_evaluate(modelsB_maml, teB_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, TEprediction=False)
        _, horacleC_maml = adapt_and_evaluate(modelsC_maml, teC_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, TEprediction=False)
        loracle_maml.append([horacleB_maml["loss"], horacleC_maml["loss"]] )
       
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
def evaluation_classification(tgen, modelA_reptile, modelsB_reptile, teB_reptile, modelsC_reptile, teC_reptile, modelA_maml, modelsB_maml, teB_maml, modelsC_maml, teC_maml, loss_fn, lr, steps=100, selection_steps=1):
    accs_reptile, accs_maml, oracle_reptile, oracle_maml = [], [], [], []
    te_preds_reptile, te_preds_maml = [], []
    te_true_reptile, te_true_maml = [], []
    id_mode = []
    for _ in range(20):
        # Get a new test task
        nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen.batch()
        id_mode.append(nnewid_label)
        
        # Adapt reptile
        _, hhistoryA_reptile = adapt_and_evaluate(modelA_reptile, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, single_model=True, acc=True, selection_steps=selection_steps)
        _, hhistoryB_reptile = adapt_and_evaluate(modelsB_reptile, teB_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, acc=True)
        _, hhistoryC_reptile = adapt_and_evaluate(modelsC_reptile, teC_reptile, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, acc=True)
        accs_reptile.append( [hhistoryA_reptile["acc"], hhistoryB_reptile["acc"], hhistoryC_reptile["acc"]] )
        te_preds_reptile.append( [hhistoryB_reptile["te_pred"], hhistoryC_reptile["te_pred"]] )
        te_true_reptile.append( [hhistoryB_reptile["te_true"], hhistoryC_reptile["te_true"]] )

        # Adapt reptile (oracle = without using task encoder) 
        _, horacleB_reptile = adapt_and_evaluate(modelsB_reptile, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, acc=True, TEprediction=False)
        _, horacleC_reptile = adapt_and_evaluate(modelsC_reptile, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, acc=True, TEprediction=False)
        oracle_reptile.append([horacleB_reptile["acc"], horacleC_reptile["acc"]] )
        
        # Adapt maml
        _, hhistoryA_maml = adapt_and_evaluate(modelA_maml, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, single_model=True, selection_steps=selection_steps, acc=True)
        _, hhistoryB_maml = adapt_and_evaluate(modelsB_maml, teB_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, acc=True)
        _, hhistoryC_maml = adapt_and_evaluate(modelsC_maml, teC_maml, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, acc=True)
        accs_maml.append( [hhistoryA_maml["acc"], hhistoryB_maml["acc"], hhistoryC_maml["acc"]] )
        te_preds_maml.append( [hhistoryB_maml["te_pred"], hhistoryC_maml["te_pred"]] )
        te_true_maml.append( [hhistoryB_maml["te_true"], hhistoryC_maml["te_true"]] )

        # Adapt maml (oracle = without using task encoder) 
        _, horacleB_maml = adapt_and_evaluate(modelsB_maml, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, acc=True, TEprediction=False)
        _, horacleC_maml = adapt_and_evaluate(modelsC_maml, _, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn, lr, steps, selection_steps=selection_steps, acc=True, TEprediction=False)
        oracle_maml.append([horacleB_maml["acc"], horacleC_maml["acc"]] )
    
    avg_accs_reptile = np.array(accs_reptile).mean(axis=0)
    avg_accs_maml = np.array(accs_maml).mean(axis=0)
    avg_oracle_reptile = np.array(oracle_reptile).mean(axis=0)
    avg_oracle_maml = np.array(oracle_maml).mean(axis=0)
    return avg_accs_reptile, te_preds_reptile, te_true_reptile, avg_oracle_reptile, avg_accs_maml, te_preds_maml, te_true_maml, avg_oracle_maml, id_mode   

# -------------------------------------------------------------------
def compare(losses, names, idx_modes, all_modes_name, save=""): #use jpg to save
    title_plot=""
    for i in idx_modes: title_plot = title_plot + all_modes_name[i] + " "
    
    fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(18, 5))
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
    if save : plt.savefig(save)
    plt.show()
    

# -------------------------------------------------------------------   
def frequency_te(te_preds, te_true, mode_labels, idx_modes, all_modes_name, n_models, method="", save=""): #use jpg to save
    # Compute accuracy
    te_acc = accuracy_score(te_preds, te_true)
    
    dictionary_pred={}
    dictionary_true={}
    for i in idx_modes:
        idxs = list(np.where(np.array(mode_labels)==i)[0])
        
        models_pred=np.array(te_preds)[idxs]
        frequency_pred = dict(Counter(models_pred))
        
        models_true=np.array(te_true)[idxs]
        frequency_true = dict(Counter(models_true))
        
        for k in range(n_models): 
            frequency_pred.setdefault(k, 0) 
            frequency_true.setdefault(k, 0) 
            
        frequency_pred = OrderedDict(sorted(frequency_pred.items()))
        frequency_true = OrderedDict(sorted(frequency_true.items()))
        
        dictionary_pred[i]=frequency_pred
        dictionary_true[i]=frequency_true
           
    cols_name = [all_modes_name[i] for i in idx_modes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4)) 
    sns.heatmap(pd.DataFrame.from_dict(dictionary_pred), xticklabels=cols_name, cmap="Blues", annot=True, cbar=False, ax=ax1)
    ax1.title.set_size(9)
    ax1.title.set_text('Prediction')
    sns.heatmap(pd.DataFrame.from_dict(dictionary_true), xticklabels=cols_name, cmap="Blues", annot=True, cbar=False, ax=ax2)
    ax2.title.set_size(9)
    ax2.title.set_text('Groundtruth')
    plt.suptitle(f"{method} Task Encoder Frequency - accuracy = {te_acc}", fontsize=11)
    if save: plt.savefig(save)
    plt.show()
    
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
    df.loc["3 models"]=[teB_accuracy_reptile, teB_accuracy_maml]
    df.loc["5 models"]=[teC_accuracy_reptile, teC_accuracy_maml]

    frequency_te(teB_preds_reptile, teB_true_reptile, eval_modes, idx_modes, all_modes_name, n_models=3, method="MReptile", save=f"{save}_MReptile_3.jpg")
    frequency_te(teC_preds_reptile, teC_true_reptile, eval_modes, idx_modes, all_modes_name, n_models=5, method="MReptile", save=f"{save}_MReptile_5.jpg")
    
    frequency_te(teB_preds_maml, teB_true_maml, eval_modes, idx_modes, all_modes_name, n_models=3, method="MMAML",save=f"{save}_MMAML_3.jpg")
    frequency_te(teC_preds_maml, teC_true_maml, eval_modes, idx_modes, all_modes_name, n_models=5, method="MMAML", save=f"{save}_MMAML_5.jpg")
    return df

# -------------------------------------------------------------------
def plot_model_knowledge(X, y, X_test, y_test, history_preds, save=""): #history_preds=history["pred"] for all the models
    fig, ax = plt.subplots(1, len(history_preds), figsize=(20, 4))
    X, y, X_test, y_test = [arr.cpu().detach() for arr in [X, y, X_test, y_test]]
    for i in range(len(history_preds)):
        ax[i].plot(X_test, history_preds[i][0], '--')
        ax[i].set_title(f"model_{i}")
    fig.suptitle("Meta-learned functions - after 0 adaptation steps")
    if save: plt.savefig(save)
    plt.show()
    
# -------------------------------------------------------------------
def plot_regression_tasks(tgen, all_modes_name, n=16):
    for i in range(n):
        X_sp, y_sp, X_qr, y_qr, label = tgen.batch()
        plt.scatter(X_sp.cpu(), y_sp.cpu())
        plt.plot(X_qr.cpu(), y_qr.cpu(), label = all_modes_name[label])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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
    ax2.plot(X_test, preds[-1], '--', label=f"After {len(preds)-1} steps")
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
        axs[i].plot(X_test, preds[i][-1], '--', label=f"After {len(preds[0])-1} steps")
    axs[0].title.set_text(f"MReptile - model {bestmodel_idx_reptile} \n Prediction probability = {round(score_idx_reptile*100, 2)}%")
    axs[1].title.set_text('Reptile')
    axs[2].title.set_text(f"MMAML - model {bestmodel_idx_maml} \n Prediction probability = {round(score_idx_maml*100, 2)}%")
    axs[3].title.set_text('MAML')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show()