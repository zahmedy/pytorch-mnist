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
hidden_size = 128

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
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (batch_size, 1, 28, 28) --> (batch_size, 784)
        x = self.fc1(x)      # (batch_size, 784) --> (batch_size, 128)
        x = self.relu(x)     # (batch_size, 128) --> (batch_size, 128)
        logits = self.fc2(x) # (batch_size, 128) --> (batch_size, 10)

        return logits


# 5. Loss & Optimizer
model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Training Loop
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

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


        