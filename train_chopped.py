import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
import os
import shutil

parser = argparse.ArgumentParser(description='VAE chopped')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 123)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set up random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def image_loader(path, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform)
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform)
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform)
    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data,
        batch_size=batch_size,
        shuffle=True
    )
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=True
    )
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True
    )
    return data_loader_sup_train, data_loader_sup_val, data_loader_unsup


# load data
train_loader, valid_loader, unsup_loader = image_loader(path='data/ssl_data_96',
                                                        batch_size=args.batch_size)


# define variational autoencoder model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # encode
        self.conv1 = nn.Conv2d(3, 9, 5)
        self.conv1_bn = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(9, 9, 5)
        self.conv2_bn = nn.BatchNorm2d(9)
        self.fc1 = nn.Linear(9 * 88 * 88, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        # latent vectors mu and logvar
        self.fc21 = nn.Linear(1024, 1024)
        self.fc22 = nn.Linear(1024, 1024)

        # decode
        self.fc2 = nn.Linear(1024, 9 * 88 * 88)
        self.fc2_bn = nn.BatchNorm1d(9 * 88 * 88)
        self.conv3 = nn.ConvTranspose2d(9, 9, 5)
        self.conv3_bn = nn.BatchNorm2d(9)
        self.conv4 = nn.ConvTranspose2d(9, 3, 5)

    def encode(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        x = self.conv2(x)
        x = F.relu(self.conv2_bn(x))
        x = x.view(-1, 9 * 88 * 88)
        x = self.fc1(x)
        h1 = F.relu(self.fc1_bn(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc2(z)
        z = F.relu(self.fc2_bn(z))
        z = z.view(-1, 9, 88, 88)
        z = self.conv3(z)
        z = F.relu(self.conv3_bn(z))
        h2 = F.relu(self.conv4(z))
        return h2

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class FineTuneModel(nn.Module):
    def __init__(self, original_model):
        super(FineTuneModel, self).__init__()
        # only the encoder of VAE (first 6 layers)
        self.features = nn.Sequential(*list(original_model.children())[:6])
        self.fc3 = nn.Linear(1024, 1000)
        self.modelName = 'chopped_net'
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        x = self.features(x)        
        x = f.view(x.size(0), -1)
        x = self.fc3(1000)
        return x

def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at checkpoint_dir + '/last.pt'. If is_best==True, also saves
    checkpoint_dir + '/best.pt'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint_dir: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint_dir, 'last.pt')
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    # else:
    #     print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best.pt'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


model = VAE().to(device)

# reload weights from restore_file if specified
restore_file = 'best_vae.pt'  # specify checkpoint to restore from or specify None
if restore_file is not None:
    restore_path = os.path.join('models', restore_file)
    print("Restoring parameters from {}".format(restore_path))
    loaded_checkpoint = utils.load_checkpoint(restore_path, model)
    print('some info on the loaded checkpoint:\n'
          '(1) average validation loss on this checkpoint is {:.4f}\n'
          '(2) it was created after {} epochs (this may not be the exact '
          'best epoch number. this is the exact best epoch number only if the last '
          'time was the only time the model was trained.)\n'
          .format(loaded_checkpoint['validation loss'],
                  loaded_checkpoint['epoch']))

# create "chopped" model
model_chopped = FineTuneModel().to(device)

# Optimizer: SGD with learning rate of 1e-2 and momentum of 0.5
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# define train method
def train(model, device, train_loader, optimizer, epoch, log_interval=1):
    # Set model to training mode
    model.train()

    # Loop through data points
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):

        # Send data and target to device
        data, target = data.to(device), target.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Pass data through model
        output = model(data)

        # Compute the negative log likelihood loss
        loss = F.nll_loss(output, target)

        # Backpropagate loss
        loss.backward()

        # Make a step with the optimizer
        optimizer.step()

        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if log:
                log_file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()


# define validation method
def validate(model, device, valid_loader):
    # Set model to evaluation mode
    model.eval()
    # Variable for the total loss
    valid_loss = 0
    # Counter for the correct predictions
    num_correct = 0

    with torch.no_grad():
        # Loop through data points
        for data, target in tqdm(valid_loader):
            # Send data to device
            data, target = data.to(device), target.to(device)

            # Pass data through model
            output = model(data)

            # Compute the negative log likelihood loss with reduction='sum' and add to total valid_loss
            valid_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            # Get predictions from the model for each data point
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

            # Add number of correct predictions to total num_correct
            num_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    # Compute the average valid loss
    avg_valid_loss = valid_loss / len(valid_loader.dataset)

    # compute validation accuracy
    valid_accuracy = 100. * num_correct / len(valid_loader.dataset)

    # print loss
    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        avg_valid_loss, num_correct, len(valid_loader.dataset),
        valid_accuracy))
    if log:
        log_file.write('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_valid_loss, num_correct, len(valid_loader.dataset),
            valid_accuracy))
    sys.stdout.flush()
    return avg_valid_loss, valid_accuracy




if __name__ == "__main__":
   # training loop
   for epoch in range(1, args.epochs + 1):
       # train model
       train(model, device, train_loader, optimizer, epoch)

       # validate model
       val_loss, val_acc = validate(model, device, valid_loader)

       is_best = val_acc >= best_val_acc

    # save weights
       utils.save_checkpoint({'epoch': epoch,
                           'validation loss': val_loss,
                           'validation accuracy': val_acc,
                           'state_dict': model.state_dict(),
                           'optim_dict': optimizer.state_dict()},
                          is_best=is_best,
                          checkpoint_dir='models_chopped')
       if is_best:
           print('- Found new best validation accuracy\n')
           if log:
               log_file.write('- Found new best validation accuracy\n\n')
               best_val_acc = val_acc





 
