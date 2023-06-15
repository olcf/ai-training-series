import torch
import torchvision
from npz_dataset import NPZDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

class ANN(torch.nn.Module):

    def __init__(self):
        super(ANN, self).__init__()

        self.linear1 = torch.nn.Linear(128, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 231)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += criterion(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    return avg_loss

class CNN2(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 231)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print("x unflat:", x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print("x shape =", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes=231):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x))) 
        x = F.relu(self.pool(self.conv2(x)))
        x = F.dropout(self.drop(x), training=self.training)
        x = x.view(-1, 32 * 32 * 24)
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)

batch_size = 4

#train_data_dir = "/gpfs/alpine/world-shared/stf218/sajal/stemdl-data/train"
#test_data_dir = "/gpfs/alpine/world-shared/stf218/sajal/stemdl-data/test"

train_data_dir = "/gpfs/wolf/world-shared/trn018/sajal/data/train"
test_data_dir = "/gpfs/wolf/world-shared/trn018/sajal/data/test"

train_dataset = NPZDataset(train_data_dir)
test_dataset = NPZDataset(test_data_dir)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

model = CNN()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
running_loss = 0.0
for epoch in range(1):
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f}')
            running_loss = 0.0
        if i == 1000:
            break;
    test(model, 'cpu', test_dataloader)

print("Finished Training")
avg_loss = running_loss / (i + 1)
print("Training set: Average loss: {:.6f}".format(avg_loss))
