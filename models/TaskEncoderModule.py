import torch
from utils import one_hot

# -------------------------------------------------------------------    
class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

# -------------------------------------------------------------------
class RegressionTaskEncoder(torch.nn.Module):
    def __init__(self, k, out_dim):
        super().__init__()
        
        self.k = k
        
        self.te_net = torch.nn.Sequential(
            torch.nn.LazyLinear(32), 
            torch.nn.ReLU(), 
            
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(), 
            
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(), 
        )
        
        self.meanLayer = torch.nn.Sequential(
            LambdaLayer(lambda x: torch.split(x, self.k)),
            LambdaLayer(lambda x: torch.stack([torch.mean(xi, dim=0, keepdim=True) for xi in x])),
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            torch.nn.BatchNorm1d(128, track_running_stats=True),
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128, track_running_stats=True),
            torch.nn.ReLU(),  
            torch.nn.Linear(128, out_dim),
            torch.nn.Softmax(dim=-1), 
        )
        

    def forward(self, X, Y): 
        # Concatenate
        x = torch.cat((X, Y), dim=1)
        # Extract features
        x = self.te_net(x)
        # Average data from the same task
        x = self.meanLayer(x)
        # Classification
        x = self.classifier(x.squeeze(dim=1))
        return x
    
    
class ClassificationTaskEncoder(torch.nn.Module):
    def __init__(self, k, in_dim=3, hidden_dim=5, out_dim=3):  # hidden_dim=n_classes, out_dim is the number of models
        super().__init__()

        self.hidden_dim = hidden_dim

        # Feature extractor for images
        self.f_net = torch.nn.Sequential(
            self.conv_block(in_dim, 32, 3, padding="same"),
            self.conv_block(32, 32, 3, padding="same"),
            self.conv_block(32, 32, 3, padding="same"),
            self.conv_block(32, 32, 3, padding="same"),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(hidden_dim),  # we don't know the in_dim since the resize can change
        )

        self.te_net = torch.nn.Sequential(
            torch.nn.LazyLinear(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )

        self.meanLayer = torch.nn.Sequential(
            LambdaLayer(lambda x: torch.split(x, k * hidden_dim)),
            LambdaLayer(lambda x: torch.stack([torch.mean(xi, dim=0, keepdim=True) for xi in x])),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.LazyLinear(8),
            torch.nn.BatchNorm1d(8, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.BatchNorm1d(16, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.Linear(16, out_dim),
            torch.nn.Softmax(dim=-1),
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
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
        # Average each task to obtain a single point per task
        x = self.meanLayer(x)
        # Apply a classifier
        x = self.classifier(x.squeeze(dim=1))
        return x
