import torch
import torch.nn as nn

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()  # Assuming input values are normalized between 0 and 1
        )
    
    def forward(self, x):
        # Pass input through the encoder
        encoded = self.encoder(x)
        # Pass encoded data through the decoder
        decoded = self.decoder(encoded)
        return decoded

# Initialize the model
input_dim = 784  # Example input dimension (e.g., for MNIST images 28x28 = 784)
hidden_dim1 = 128
hidden_dim2 = 64
latent_dim = 32

model = Autoencoder(input_dim, hidden_dim1, hidden_dim2, latent_dim)

# Print the model structure
print(model)

# Example usage
# Input data
x = torch.randn((16, input_dim))  # Batch size of 16
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
