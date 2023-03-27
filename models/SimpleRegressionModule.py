import torch
    
# -------------------------------------------------------------------
class RegressionModule(torch.nn.Module): # Same architecture in MAML and Reptile paper
    def __init__(self):
        super().__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 40), 
            torch.nn.BatchNorm1d(40, track_running_stats=False), 
            torch.nn.ReLU(), 
            
            torch.nn.Linear(40, 40), 
            torch.nn.BatchNorm1d(40, track_running_stats=False), 
            torch.nn.ReLU(), 
            
            torch.nn.Linear(40, 1),
        )

    def forward(self, x):
        return self.net(x)
    
# # -------------------------------------------------------------------    
# class RegressionModule(torch.nn.Module): # Tuned architecture on single Reptile
#     def __init__(self):
#         super().__init__()
        
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(1, 300), 
#             torch.nn.BatchNorm1d(300, track_running_stats=False), 
#             torch.nn.ReLU(), 
            
#             torch.nn.Linear(300, 100), 
#             torch.nn.BatchNorm1d(100, track_running_stats=False), 
#             torch.nn.ReLU(), 
            
#             torch.nn.Linear(100, 130), 
#             torch.nn.BatchNorm1d(130, track_running_stats=False), 
#             torch.nn.ReLU(), 
            
#             torch.nn.Linear(130, 1),
#         )

#     def forward(self, x):
#         return self.net(x)