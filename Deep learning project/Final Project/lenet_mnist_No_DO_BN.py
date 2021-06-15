#swapnil_kamate_deep_learning_project
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import time
from collections import OrderedDict
from torchvision.transforms import ToTensor

# Preparing for Data
print('==> Preparing data..')

#classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.convnet = nn.Sequential(OrderedDict([
            ('c1',nn.Conv2d(1,6,kernel_size=(5,5))),         
            ('relu1',nn.ReLU()),                                    
            ('s2',nn.MaxPool2d(kernel_size=(2,2), stride=2)),       
            ('c3',nn.Conv2d(6,16,kernel_size=(5,5))),               
            ('relu3',nn.ReLU()),                                    
            ('s4',nn.MaxPool2d(kernel_size=(2,2), stride=2)),       
            ('c5',nn.Conv2d(16,120,kernel_size=(4,4))),             
            ('relu5',nn.ReLU())                                     
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6',nn.Linear(120,84)),                               
            ('relu6',nn.ReLU()),                               
            ('f7',nn.Linear(84,10)),                              
            ('sig7',nn.LogSoftmax(dim=-1))                         
        ]))


    def forward(self, x):

        out = self.convnet(x)      
        # Flattening the data to fed it to FC. Linear Layers                    
        out = out.view(out.size(0), -1)                 
        out = self.fc(out)                            
    
        return out



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()                           
        output  = model(data)                           
        criterion = nn.CrossEntropyLoss()               
        loss = criterion(output,target)               
        loss.backward()                                 
        optimizer.step()                                

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test( model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    time0 = time.time()
    # Training settings
    batch_size = 128
    epochs = 25
    lr = 0.05
    no_cuda = True
    save_model = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(100)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    trainset = torchvision.datasets.MNIST(root='./train', train=True, download=True, transform=ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./test', train=False, download=True, transform=ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, epochs + 1):
        train( model, device, train_loader, optimizer, epoch)
        test( model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(),"mnist_lenet.pt")
    time1 = time.time() 
    print ('Traning and Testing total excution time is: %s seconds ' % (time1-time0))   
if __name__ == '__main__':
    main()