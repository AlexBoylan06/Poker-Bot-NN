import torch
import torch.nn as nn
import torch.optim as optim

class PokerNN(nn.Module):
    def __init__(self, input_size=52, output_size=10):
        super(PokerNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size),
            nn.Softmax(dim=1)  # Output as probabilities
        )

    def forward(self, x):
        return self.model(x)

# Example usage:
if __name__ == "__main__":
    input_size = 52
    output_size = 10

    model = PokerNN(input_size, output_size)
    print(model)

    # Example input (batch size = 1, 52 features)
    sample_input = torch.randn(1, input_size)
    output = model(sample_input)
    print(output)

