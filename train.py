import torch
import torch.nn as nn
import torch.optim as optim
from data import train_loader, test_loader
from config import hidden_size, learning_rate, num_epochs
from model import SimpleMLP


# 5. Loss & Optimizer
model = SimpleMLP(hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Training Loop
def main(num_epochs,train_loader, test_loader):
    losses = []
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            # convert to python number from scaler tensor 
            # and accumlate all losses for all epochs (per batch graph)
            losses.append(loss.item())
            # Accumlate all losses for all batches (per epoch graph)
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

if __name__ == "__main__":
    main(num_epochs, train_loader, test_loader)