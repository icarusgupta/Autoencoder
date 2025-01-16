import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Define the training function
def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            # Move data to the device
            inputs, _ = batch  # `_` ignores the labels
            inputs = inputs.view(inputs.size(0), -1).to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Load dataset (e.g., MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize data to [-1, 1]
])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Hyperparameters
input_dim = 784  # 28x28 flattened
hidden_dim1 = 128
hidden_dim2 = 64
latent_dim = 32
learning_rate = 0.001
num_epochs = 10

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(input_dim, hidden_dim1, hidden_dim2, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the autoencoder
train_autoencoder(model, dataloader, criterion, optimizer, num_epochs, device)

# Save the trained model
torch.save(model.state_dict(), "autoencoder.pth")
print("Model saved as 'autoencoder.pth'")
