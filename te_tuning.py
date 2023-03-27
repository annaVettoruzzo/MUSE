import warnings

warnings.filterwarnings("ignore")

import torch, random, numpy as np, pandas as pd, pickle
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from dataset import ClassificationTaskGenerator
from metalearning import MultiReptile, MultiMAML, TaskEncoderTrainer, adapt_and_evaluate
from models import ClassificationModule, ClassificationTaskEncoder
from utils import DEVICE, write_in_file, evaluation_classification, plot_compare, te_evaluation, frequency_te

# For reproducibility
torch.random.manual_seed(5)
np.random.seed(5)
random.seed(5)

all_modes_name = ["FGVC_Aircraft", "FGVCx_Fungi", "CUB_Bird", "Omniglot", "CifarFS", "MiniImageNet"]
idx_modes = [0,1,2]

path_dir = "Results/classification_simple/tuning/soft_adaptive"

################# SET PARAMETERS #####################
from params_classification_simple import *

###################### DATASET ##########################
print("LOAD DATASET")
# To sample training tasks randomly (for meta-training)
tgen = ClassificationTaskGenerator(n_classes, k, q, idx_modes, split="train", resize=84)
# To sample new evaluation tasks (for meta-testing)
tgen_eval = ClassificationTaskGenerator(n_classes, k, q, idx_modes, split="test", resize=84)

modelA_maml = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE)]
for model in modelA_maml: model.load_state_dict(torch.load(f"{path_dir}/it1/maml/model_0.pt"))

modelsB_maml = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE) for _ in range(3)]
for i, model in enumerate(modelsB_maml): model.load_state_dict(torch.load(f"{path_dir}/it1/maml/3models_{i}.pt"))

modelsC_maml = [ClassificationModule(in_dim, n_classes=n_classes).to(DEVICE) for _ in range(5)]
for i, model in enumerate(modelsC_maml): model.load_state_dict(torch.load(f"{path_dir}/it1/maml/5models_{i}.pt"))

################# TUNING ###############################
import optuna
from utils import one_hot

te_steps = 1000

from utils import one_hot

# -------------------------------------------------------------------
class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            # torch.nn.BatchNorm2d(out_channels, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )


def define_model(trial, in_dim=3, hidden_dim=5, out_dim=3):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []

    layers.append(conv_block(in_dim, 32, 3, padding="same"))
    in_features = 32

    for i in range(n_layers - 1):
        layers.append(conv_block(in_features, 32, 3, padding="same"))
        in_features = 32
    layers.append(torch.nn.Flatten())
    layers.append(torch.nn.LazyLinear(hidden_dim))

    ###############
    n_layers_te = trial.suggest_int("n_layers_te", 1, 4)
    layers_te = []

    out_features = trial.suggest_categorical("n_units_in_te", [8, 16, 32, 64])
    layers_te.append(torch.nn.LazyLinear(out_features))
    layers_te.append(torch.nn.ReLU())
    in_features = out_features

    for i in range(n_layers_te - 1):
        out_features = trial.suggest_categorical("n_units_l{}_te".format(i), [8, 16, 32, 64])
        layers_te.append(torch.nn.Linear(in_features, out_features))
        layers_te.append(torch.nn.ReLU())
        in_features = out_features

    ###############
    n_layers_class = trial.suggest_int("n_layers_class", 1, 4)
    layers_class = []

    out_features = trial.suggest_categorical("n_unitsclass_in", [8, 16, 32, 64])
    layers_class.append(torch.nn.LazyLinear(out_features))
    layers_class.append(torch.nn.BatchNorm1d(out_features, track_running_stats=True))
    layers_class.append(torch.nn.ReLU())
    in_features = out_features

    for i in range(n_layers_class - 1):
        out_features = trial.suggest_categorical("n_unitsclass_l{}".format(i), [8, 16, 32, 64])
        layers_class.append(torch.nn.Linear(in_features, out_features))
        layers_class.append(torch.nn.BatchNorm1d(out_features, track_running_stats=True))
        layers_class.append(torch.nn.ReLU())

        in_features = out_features
    layers_class.append(torch.nn.Linear(out_features, out_dim))
    layers_class.append(torch.nn.Softmax(dim=-1))

    return torch.nn.Sequential(*layers), torch.nn.Sequential(*layers_te), torch.nn.Sequential(*layers_class)


class ClassificationTaskEncoder(torch.nn.Module):
    def __init__(self, trial, in_dim=3, hidden_dim=5, out_dim=3):  # hidden_dim=n_classes, out_dim is the number of models
        super().__init__()

        self.hidden_dim = hidden_dim

        self.f_net, self.te_net, self.classifier = define_model(trial, out_dim=out_dim)

        self.meanLayer = torch.nn.Sequential(
            LambdaLayer(lambda x: torch.split(x, 5 * hidden_dim)),
            LambdaLayer(lambda x: torch.stack([torch.mean(xi, dim=0, keepdim=True) for xi in x])),
        )

    def forward(self, X, Y):
        # Tranform X into am embedded feature vector
        x = self.f_net(X)
        # Convert into a one-hot vector
        y = one_hot(Y, self.hidden_dim)
        # Concatenate x and y
        x = torch.cat((x, y), dim=1)
        # Extract features
        x = self.te_net(x)
        # Average eact task to obtain a single point per task
        x = self.meanLayer(x)
        # Apply a classifier
        x = self.classifier(x.squeeze(dim=1))
        return x

def objective(trial):
    # Generate the model.
    teB_model = ClassificationTaskEncoder(trial, in_dim, n_classes, 3).to(DEVICE)
    teC_model = ClassificationTaskEncoder(trial, in_dim, n_classes, 5).to(DEVICE)

    te_lr = trial.suggest_uniform("lr", 1e-5, 1e-1)

    TaskEncoderTrainer(teB_model, modelsB_maml, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps,
                                                                                selection_steps, lr)
    TaskEncoderTrainer(teC_model, modelsC_maml, loss_fn, te_loss_fn, te_lr).fit(tgen, te_batchsize, te_steps,
                                                                                selection_steps, lr)

    teB_true, teB_pred, teC_true, teC_pred = [], [], [], []
    for i in range(10):
        # Get a new test task
        nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, nnewid_label = tgen_eval.batch()

        # Adapt the models A, B, C, D, to the new task
        _, hhistoryB = adapt_and_evaluate(modelsB_maml, teB_model, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn,
                                          lr, steps=50, selection_steps=selection_steps, acc=True)
        _, hhistoryC = adapt_and_evaluate(modelsC_maml, teC_model, nnewX_sp, nnewy_sp, nnewX_qr, nnewy_qr, loss_fn,
                                          lr, steps=50, selection_steps=selection_steps, acc=True)

        teB_true.append(hhistoryB["te_true"])
        teB_pred.append(hhistoryB["te_pred"])

        teC_true.append(hhistoryC["te_true"])
        teC_pred.append(hhistoryC["te_pred"])
    print(teB_true)
    print("\n")
    print(teB_pred)
    teB_acc = accuracy_score(teB_true, teB_pred)
    teC_acc = accuracy_score(teC_true, teC_pred)
    return (teB_acc + teC_acc) / 2


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
