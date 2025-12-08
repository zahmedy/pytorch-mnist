import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



# 2. Hyperparameters
batch_size = 32
learning_rate = 0.01
num_epochs = 10
hidden_size = 64

# 3. Dataset & DataLoaders
mean = 0.1307
std = 0.3081
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((mean,), (std,),)]
    )

train_dataset = datasets.MNIST("./data", 
                               train=True, 
                               download=True, 
                               transform=transform)

test_dataset = datasets.MNIST("./data",
                              train=False,
                              download=True,
                              transform=transform)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# 4. MLP Model
class SimpleMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (batch_size, 1, 28, 28) --> (batch_size, 784)
        x = self.fc1(x)      # (batch_size, 784) --> (batch_size, hidden_size)
        x = self.relu(x)     # (batch_size, hidden_size) --> (batch_size, hidden_size)
        x = self.fc2(x)      # (batch_size, hidden_size) --> (batch_size, 32)
        x = self.relu(x)     # (batch_size, 32) --> (batch_size, 32)
        logits = self.fc3(x) # (batch_size, 32) --> (batch_size, 10)

        return logits


# 5. Loss & Optimizer
model = SimpleMLP(hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Training Loop
losses = []
epoch_losses = []
for epoch in range(num_epochs):
    running_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)

# 7. Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        logits = model(images)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")

# visualize the loss curve 
import matplotlib.pyplot as plt

plt.plot(epoch_losses)
plt.title("Epoch Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Batch Number")
plt.ylabel("Loss")
plt.show()

# 8. Save model
torch.save(model.state_dict(), "mnist_mlp.pth")
