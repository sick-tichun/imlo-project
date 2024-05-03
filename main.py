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
torch.backends.cudnn.benchmark = True

#defining a transformer for our training dataset to randomise
#rotations, aspect ratios etc of the image, so the model isnt thrown off by these aspects
t_transform = transforms.Compose([
    transforms.RandomRotation(30), #rotates the images randomly
    transforms.RandomResizedCrop(size=224), #resizes/crops image
    transforms.RandomHorizontalFlip(), #flips
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #squishes data into a range wich is vital to a CNN network to performing
])
v_transfrom = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #squishes data into a range wich is vital to a CNN network to performing
    ])

#normalize to be possibly adjusted later, these are placehld values


#loading our datast (with transformation applied)
train_data = datasets.Flowers102(root='data/train',download=True, split='train', transform=t_transform, target_transform = Lambda(lambda label: torch.zeros(1020, dtype=torch.float).scatter_(dim=0, index=torch.tensor(label), value=1)))
valid_data = datasets.Flowers102(root='data/valid',download=True, split='val', transform=v_transfrom)
test_data = datasets.Flowers102(root='data/test',download=True, split='test')

batch_size = 200

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

#implementing the NN given in the lecture (this will be changed to a convelutional neural network later)

class My_NN(nn.Module):
    def __init__(self, in_feature, hidden_layers, out_features, activation_function = F.relu):
        super().__init__()
        if len(hidden_layers) < 1:
            raise Exception("gotta have more than 1 hidden layer")
        
        self.layers = []
        self.layers.append(nn.Linear(in_feature, hidden_layers[0]))
        self.add_module('input layer', self.layers[0])

        for i in range (1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.add_module('hidden_layer_{i}', self.layers[i])

        self.out = nn.Linear(hidden_layers[-1], out_features)

        self.activation_function = activation_function
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation_function(self.layers[i](x))
        x = self.out(x)
        return x

print(len(train_data._labels))
classifier = My_NN(in_feature=224, hidden_layers=[32, 16], out_features=len(train_data._labels)).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
epochs = 20


losses = []
for i in range(epochs):
    for batch, label in train_loader:
        batch, label = batch.to(device), label.to(device)
        batch = torch.argmax(batch, dim=1)
        outputs = classifier.forward(batch)
        print(label)
        loss = loss_func(outputs, label)
    losses.append(loss)
    if i % 10 == 0:
        print('epoch' + str(i) + str(loss))

    optimizer.zero_grad()
    loss.backwards()
    optimizer.step()

