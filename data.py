
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from config import mean, std, batch_size

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