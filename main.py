#importing hte neccecary libraries
import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data

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
train_data = datasets.Flowers102(root='data/train',download=True, split='train', transform=t_transform)
valid_data = datasets.Flowers102(root='data/valid',download=True, split='val', transform=v_transfrom)
test_data = datasets.Flowers102(root='data/test',download=True, split='test')

batch_size = 10

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)



