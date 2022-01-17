from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layers[0]),
            nn.BatchNorm1d(self.hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[0], self.hidden_layers[1]),
            nn.BatchNorm1d(self.hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[1], self.hidden_layers[2]),
            nn.BatchNorm1d(self.hidden_layers[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[2], self.output_size),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)
        return self.model(x)