#importing hte neccecary libraries
import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


#defining a transformer for our training dataset to randomise
#rotations, aspect ratios etc of the image, so the model isnt thrown off by these aspects
size = 224
t_transform = transforms.Compose([
    transforms.RandomRotation(30), #rotates the images randomly
    transforms.RandomResizedCrop(size=size), #resizes/crops image
    transforms.RandomHorizontalFlip(), #flips
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #squishes data into a range wich is vital to a CNN network to performing 
 ]) 
v_transfrom = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
#Normalise used precalculated values from https://howieko.com/post/classifying_flowers_pytorch/


#loading our datast (with transformation applied)
train_data = datasets.Flowers102(root='data/train',download=True, split='train', transform=t_transform)
valid_data = datasets.Flowers102(root='data/valid',download=True, split='val', transform=v_transfrom)
test_data = datasets.Flowers102(root='data/test',download=True, split='test', transform=test_transform)

batch_size = 16

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

#implementing the NN given in the lecture (this will be changed to a convelutional neural network later)

class network_old(nn.Module):
    def __init__(self, in_feature, hidden_layers, out_features):
        super().__init__()
        if len(hidden_layers) < 1:
            raise Exception("gotta have more than 1 hidden layer")
        self.flatten = nn.Flatten()
        
        self.layers = []
        self.layers.append(nn.Linear(in_feature, hidden_layers[0]))
        self.add_module('input layer', self.layers[0])

        for i in range (1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.add_module('hidden_layer_{i}', self.layers[i])

        self.out = nn.Linear(hidden_layers[-1], out_features)
        self.add_module('output layer', self.out)
        self.activation_function = nn.ReLU()
        print(self.layers, self.out)

    def forward(self, x):
        x = self.flatten(x)
        for i in range(len(self.layers)):
            x = self.activation_function(self.layers[i](x))
        x = self.out(x).to(device)
        return x

#implementing a CNN
class Cnetwork(nn.Module):
    def __init__(self, in_channels = 3, out_features=102, size = 224):
        super().__init__()
        self.out_features = out_features
        self.stack = nn.Sequential(
            # "same" convolution layer (ie input feat = output feat)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            #reduces the size of the image by half
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )
        self.fully_con = nn.Sequential(
            nn.Linear(14336, self.out_features)
        )
        
    def forward(self, x):
        x = self.stack(x)
        x.reshape(x.size(0), -1).to(device)
        x = self.fully_con(x)
        return x


#classifier = network_old(in_feature=size*size*3, hidden_layers=[2048, 512], out_features=102).to(device)
classifier = torch.load('model.pth')
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

epochs = 10

def train(loader, model=classifier, loss_func = loss_func, optimizer = optimizer):
    model.train()
    print('asdasdaasdd asasd'+str(next(model.parameters()).is_cuda))
    size = len(loader.dataset)
    for batch, (X, y) in enumerate(loader):
       
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_func(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 80 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    train(train_loader, classifier)
    test(valid_loader, classifier, loss_func)
    if i % 2 == 0 and i != 0:
      torch.save(classifier.state_dict(), 'modeltemp.pth')   
print("Done!")
test(test_loader, classifier, loss_func)
torch.save(classifier.state_dict(), 'model.pth')
