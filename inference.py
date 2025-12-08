
from minst_mlp import SimpleMLP, test_dataset
import torch

loaded_model = SimpleMLP(64)
loaded_model.load_state_dict(torch.load("mnist_mlp.pth"))
loaded_model.eval()

image, label = test_dataset[0]      # image: (1, 28, 28)
image = image.unsqueeze(0)          # (1, 1, 28, 28) -> add batch dim

with torch.no_grad():
    logits = loaded_model(image)
    pred = logits.argmax(dim=1).item()

    print("True label:", label)
    print("Predicted:", pred)

