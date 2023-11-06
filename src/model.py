from torch import nn

class Network(nn.Module):
    N_DOGS = 6
    DOG_FEATURES = 3
    GLOBAL_FEATURES = 1
    LAYER_1_NODES = 64
    LAYER_2_NODES = 64

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._layers = nn.Sequential(
            nn.Linear(
                self.N_DOGS * self.DOG_FEATURES + self.GLOBAL_FEATURES,
                self.LAYER_1_NODES,
            ),
            nn.ReLU(),
            nn.Linear(self.LAYER_1_NODES, self.LAYER_2_NODES),
            nn.ReLU(),
            nn.Linear(self.LAYER_2_NODES, self.N_DOGS),
            nn.Softmax(dim=0),
        )
        

    def forward(self, x):
        logits = self._layers(x)
        return logits
