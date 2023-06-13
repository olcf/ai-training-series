import torch
import torchvision
from npz_dataset import NPZDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import warnings
warnings.filterwarnings("ignore")
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def setup_ddp(backend):
    """"Initialize DDP"""
    import subprocess
    try:
        get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(os.environ['LSB_DJOB_HOSTFILE'])
        master_addr = str(subprocess.check_output(get_master, shell=True))[2:-3]
        master_port = "29500"
        world_size = os.environ['OMPI_COMM_WORLD_SIZE']
        world_rank = os.environ['OMPI_COMM_WORLD_RANK']
    except KeyError:
        print("DDP has to be initialized within a job")
        sys.exit(1)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = world_size
    os.environ['RANK'] = world_rank
    dist.init_process_group(backend=backend, rank=int(world_rank), world_size=int(world_size))

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
            data, target = data.to(world_rank % 6), target.to(world_rank % 6)

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


setup_ddp("nccl")
world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

batch_size = 32

train_data_dir = "/gpfs/alpine/world-shared/stf218/sajal/stemdl-data/train"
test_data_dir = "/gpfs/alpine/world-shared/stf218/sajal/stemdl-data/test"

train_dataset = NPZDataset(train_data_dir)
test_dataset = NPZDataset(test_data_dir)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = "cuda"
#model = CNN()
num_classes = 231
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(world_rank % 6)
model = DDP(model, device_ids = [world_rank % 6])
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
running_loss = 0.0
for epoch in range(1):
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(world_rank % 6), labels.to(world_rank % 6)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f}')
            running_loss = 0.0
        if i == 100:
            break;
    test(model, device, test_dataloader)

print("Finished Training")
avg_loss = running_loss / (i + 1)
print("Training set: Average loss: {:.6f}".format(avg_loss))
