import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import transforms, datasets
from tqdm import tqdm

parser = argparse.ArgumentParser(description='VAE')
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
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 24 * 24, 120)
        self.fc1_bn = nn.BatchNorm1d(120)

        # latent vectors mu and logvar
        self.fc21 = nn.Linear(120, 400)
        self.fc22 = nn.Linear(120, 400)

        # decode
        self.fc2 = nn.Linear(400, 16 * 24 * 24)
        self.fc2_bn = nn.BatchNorm1d(16 * 24 * 24)
        self.conv3 = nn.ConvTranspose2d(16, 6, 5)
        self.conv3_bn = nn.BatchNorm2d(6)
        self.conv4 = nn.ConvTranspose2d(6, 3, 5)

    def encode(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        x = self.conv2(x)
        x = F.relu(self.conv2_bn(x))
        x = x.view(-1, 16 * 24 * 24)
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
        z = z.view(-1, 16, 24, 24)
        z = self.conv3(z)
        z = F.relu(self.conv3_bn(z))
        h2 = F.relu(self.conv4(z))
        return h2

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


# define train method
def train(epoch, data_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in tqdm(enumerate(data_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(data_loader.dataset)))


# define validate method
def validate(epoch, data_loader):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, (data, _) in tqdm(enumerate(data_loader)):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            valid_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(data.shape[0], 3, 32, 32)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    valid_loss /= len(data_loader.dataset)
    print('====> Valid set loss: {:.4f}'.format(valid_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch, data_loader=unsup_loader)
        validate(epoch, data_loader=valid_loader)
