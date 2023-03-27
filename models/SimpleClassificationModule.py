import torch


# -------------------------------------------------------------------
class ClassificationModule(torch.nn.Module):
    def __init__(self, in_dim=3, n_filters=32, hidden_dim=800, n_classes=5):
        super().__init__()

        self.conv_net = torch.nn.Sequential(
            self.conv_block(in_dim, n_filters, 3, padding="same"),
            self.conv_block(n_filters, n_filters, 3, padding="same"),
            self.conv_block(n_filters, n_filters, 3, padding="same"),
            self.conv_block(n_filters, n_filters, 3, padding="same"),
            torch.nn.Flatten(),
        )

        self.fc_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_classes)
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc_net(x)
        return x


# -------------------------------------------------------------------
class ClassificationModule2(torch.nn.Module):
    def __init__(self, in_dim=3, n_filters=32, hidden_dim=800*8, n_classes=5):
        super().__init__()

        self.conv_net = torch.nn.Sequential(
            self.conv_block(in_dim, n_filters, 3, padding="same"),
            self.conv_block(n_filters, n_filters*2, 3, padding="same"),
            self.conv_block(n_filters*2, n_filters*4, 3, padding="same"),
            self.conv_block(n_filters*4, n_filters*8, 3, padding="same"),
            torch.nn.Flatten(),
        )

        self.fc_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_classes)
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc_net(x)
        return x
