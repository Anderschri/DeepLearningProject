"""
Semi-supervised Learning with Deep Generative Models
https://arxiv.org/pdf/1406.5298.pdf
"""

import os
import argparse
from tqdm import tqdm
import pprint
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

# PixelCNN++
from pcnnpp_utils import *
from pcnnpp_model import *

parser = argparse.ArgumentParser()

# actions
parser.add_argument('--dataset', type=str, default='mnist', help='Accepts mnist or cifar10')
parser.add_argument('--train', action='store_true', help='Train a new or restored model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a model.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--vis_styles', action='store_true', help='Visualize styles manifold.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--calc_nll', action='store_true', help='Calculate NLL?')
parser.add_argument('--generate_samples', action='store_true', help='Sample z and y according to distributions and decode?')

# file paths
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--data_dir', default='./data/', help='Location of dataset.')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')

# model parameters
parser.add_argument('--z_depth', type=tuple, default=1, help='Number of channels in latent space (e.g. 1).')
# parser.add_argument('--image_dims', type=tuple, default=(1,28,28), help='Dimensions of a single datapoint (e.g. (1,28,28) for MNIST).')
parser.add_argument('--z_dim', type=int, default=8, help='Size of the latent representation.')
parser.add_argument('--y_dim', type=int, default=10, help='Size of the labels / output.')
parser.add_argument('--hidden_dim', type=int, default=500, help='Size of the hidden layer.')
parser.add_argument('--mixture_comps', type=int, default=1, help='Number of mixture components')

# training params
parser.add_argument('--n_labeled', type=int, default=3000, help='Number of labeled training examples in the dataset')
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--alpha', type=float, default=0.1, help='Classifier loss multiplier controlling generative vs. discriminative learning.')


# --------------------
# Data
# --------------------

# create semi-supervised datasets of labeled and unlabeled data with equal number of labels from each class
def create_semisupervised_datasets(dataset, n_labeled, dataname):
    # note this is only relevant for training the model
    assert dataset.train == True, 'Dataset must be the training set; assure dataset.train = True.'

    # compile new x and y and replace the dataset.train_data and train_labels with the 
    #x = dataset.train_data
    #y = dataset.train_labels
    x = dataset.data
    y = dataset.targets

    if dataname=='cifar10':
        y = torch.LongTensor(y) #Anton
    n_x = x.shape[0]
    n_classes = len(torch.unique(y))

    assert n_labeled % n_classes == 0, 'n_labeld not divisible by n_classes; cannot assure class balance.'
    n_labeled_per_class = n_labeled // n_classes

    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes

    for i in range(n_classes):
        idxs = (y == i).nonzero().data.numpy()
        np.random.shuffle(idxs)

        x_labeled[i] = x[idxs][:n_labeled_per_class]
        y_labeled[i] = y[idxs][:n_labeled_per_class]
        x_unlabeled[i] = x[idxs][n_labeled_per_class:]
        y_unlabeled[i] = y[idxs][n_labeled_per_class:]

    # construct new labeled and unlabeled datasets
    ### Edit: train_data and train_labels edited to .data and .targets
    labeled_dataset = copy.deepcopy(dataset)
    if dataname == 'mnist':
        labeled_dataset.data = torch.cat(x_labeled, dim=0).squeeze()
    elif dataname == 'cifar10':
        labeled_dataset.data = np.reshape(x_labeled,(-1,x.shape[1],x.shape[2],x.shape[3]))
    #labeled_dataset.train_data = torch.cat(x_labeled, dim=0).squeeze()
    labeled_dataset.targets = torch.cat(y_labeled, dim=0)

    unlabeled_dataset = copy.deepcopy(dataset)
    if dataname == 'mnist':
        unlabeled_dataset.data = torch.cat(x_unlabeled, dim=0).squeeze()
    elif dataname == 'cifar10':
        unlabeled_dataset.data = np.reshape(x_unlabeled,(-1,x.shape[1],x.shape[2],x.shape[3]))
    unlabeled_dataset.targets = torch.cat(y_unlabeled, dim=0)

    del dataset

    return labeled_dataset, unlabeled_dataset


def fetch_dataloaders(args):
    assert args.n_labeled != None, 'Must provide n_labeled number to split dataset.'

    transforms = T.Compose([T.ToTensor()])
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device.type is 'cuda' else {}

    def get_dataset(dataset, train):
         if dataset=='mnist':
            return MNIST(root=args.data_dir, train=train, transform=transforms, download=True)
         elif dataset=='cifar10':
            return CIFAR10(root=args.data_dir, train=train, transform=transforms, download=True)


    def get_dl(dataset):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=dataset.train, drop_last=True, **kwargs)

    test_dataset = get_dataset(dataset=args.dataset, train=False)
    train_dataset = get_dataset(dataset=args.dataset, train=True)
    labeled_dataset, unlabeled_dataset = create_semisupervised_datasets(train_dataset, args.n_labeled, args.dataset)

    return get_dl(labeled_dataset), get_dl(unlabeled_dataset), get_dl(test_dataset)


def one_hot(x, label_size):
    out = torch.zeros(len(x), label_size).to(x.device)
    out[torch.arange(len(x)), x.squeeze()] = 1
    return out

# --------------------
# Model
# --------------------

class SSVAE(nn.Module):
    """
    Data model (SSL paper eq 2):
        p(y) = Cat(y|pi)
        p(z) = Normal(z|0,1)
        p(x|y,z) = f(x; z,y,theta)

    Recognition model / approximate posterior q_phi (SSL paper eq 4):
        q(y|x) = Cat(y|pi_phi(x))
        q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x)))


    """
    def __init__(self, args):
        super().__init__()
        C, H, W = args.image_dims
        x_dim = C * H * W


        # --------------------
        # p model for z -- PixelCNN++
        # --------------------

        self.p_z = PixelCNN(nr_resnet=1, nr_filters=10, input_channels=args.z_depth, nr_logistic_mix=args.mixture_comps) # Input_channels = 1 if N x N x 1 grid, 3 if N x N x 3 grid
            #(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
            #input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)

        # D.Normal(torch.tensor(0., device=args.device), torch.tensor(1., device=args.device))

        # --------------------
        # p model -- SSL paper generative semi supervised model M2
        # --------------------

        self.p_y = D.OneHotCategorical(probs=1 / args.y_dim * torch.ones(1,args.y_dim, device=args.device))
        # self.p_z = D.Normal(torch.tensor(0., device=args.device), torch.tensor(1., device=args.device)) # Old


        # parametrized data likelihood p(x|y,z)
        self.decoder = nn.Sequential(#nn.Dropout(0.5),
                                     nn.Linear(args.z_depth*args.z_dim**2 + args.y_dim, args.hidden_dim),
                                     nn.BatchNorm1d(args.hidden_dim),
                                     nn.Softplus(),
                                     nn.Linear(args.hidden_dim, args.hidden_dim),
                                     nn.BatchNorm1d(args.hidden_dim),
                                     nn.Softplus(),
                                     # nn.Dropout(0.5),
                                     nn.Linear(args.hidden_dim, x_dim),
                                     # nn.BatchNorm1d(x_dim),
                                     nn.Softplus())

        #self.decoder_cnn = nn.Sequential(
        #                               #nn.Dropout(0.5),
        #                               nn.Conv2d(in_channels=args.image_dims[0], out_channels=10, kernel_size=3, stride=1, padding=1), ### <----------- EVT TILFØJ FLERE CHANNELS
        #                               nn.BatchNorm2d(10), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
        #                               nn.Softplus(),
        #                                nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
        #                               nn.BatchNorm2d(20), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
        #                               nn.Softplus(),
        #                               # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #                               # nn.Dropout(0.4),
        #                               nn.Conv2d(in_channels=20, out_channels=args.image_dims[0], kernel_size=3, stride=1, padding=1),
        #                               #nn.BatchNorm2d(args.image_dims[0]), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
        #                               #nn.Softplus()
        #                                )

        # Transposed Conv test
        # Before: 1 -> 10 -> 20 -> 1
        self.decoder_tcnn = nn.Sequential(
                                       # nn.Dropout(0.5),
                                       nn.ConvTranspose2d(in_channels=args.image_dims[0], out_channels=10, kernel_size=3, stride=1, padding=1), ### <----------- EVT TILFØJ FLERE CHANNELS
                                       nn.BatchNorm2d(10), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
                                       nn.Softplus(),
                                         nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2),
                                       nn.Softplus(),
                                        nn.ConvTranspose2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(20), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
                                       nn.Softplus(),
                                       # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                       # nn.Dropout(0.4),
                                       nn.ConvTranspose2d(in_channels=20, out_channels=args.image_dims[0], kernel_size=3, stride=1, padding=1),
                                       #nn.BatchNorm2d(args.image_dims[0]), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
                                       #nn.Softplus()
                                        )

        # --------------------
        # q model -- SSL paper eq 4
        # --------------------

        # parametrized q(y|x) = Cat(y|pi_phi(x)) -- outputs parametrization of categorical distribution
        #before: 1 -> 10 -> 20
        self.encoder_y_cnn = nn.Sequential(
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(in_channels=args.image_dims[0], out_channels=10, kernel_size=5, stride=1, padding=2),
                                       nn.BatchNorm2d(10), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
                                       nn.Softplus(),
                                       nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                       # nn.Dropout(0.4),
                                       nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
                                       nn.Softplus(),
                                       nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(20), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
                                       nn.Softplus(),
                                       nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                                        )

        self.encoder_y = nn.Sequential(# nn.Dropout(0.5),
                                       nn.Linear(20*H//4*W//4, args.hidden_dim), # x_dim i stedet for
                                       # nn.BatchNorm1d(args.hidden_dim),
                                       nn.Softplus(),
                                       #nn.Linear(args.hidden_dim, args.hidden_dim),
                                       #nn.Softplus(),
                                       nn.Linear(args.hidden_dim, args.hidden_dim),
                                       # nn.BatchNorm1d(args.hidden_dim),
                                       nn.Softplus(),
                                       # nn.Dropout(0.5),
                                       nn.Linear(args.hidden_dim, args.y_dim))

        # parametrized q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- output parametrizations for mean and diagonal variance of a Normal distribution
        #before: 1 -> 10 -> 20
        self.encoder_z_cnn = nn.Sequential(
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(in_channels=args.image_dims[0], out_channels=10, kernel_size=5, stride=1, padding=2),
                                       nn.BatchNorm2d(10), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
                                       nn.Softplus(),
                                       nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2),
                                       #nn.BatchNorm2d(20), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
                                       nn.Softplus(),
                                       nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                       # nn.Dropout(0.4),
                                       nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(20), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
                                       nn.Softplus(),
                                       #nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
                                       #nn.BatchNorm2d(20), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
                                       #nn.Softplus(),

                                       #nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                       #nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
                                       ##nn.BatchNorm2d(20), # batch normalization before activation function as suggested in Ioffe and Szegedy 2015
                                       #nn.Softplus(),
                                       nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.encoder_z = nn.Sequential(# nn.Dropout(0.5),
                                       nn.Linear(20*H//4*W//4 + args.y_dim, args.hidden_dim), # x_dim + args.y_dim
                                       # nn.BatchNorm1d(args.hidden_dim),
                                       nn.Softplus(),
                                       nn.Linear(args.hidden_dim, args.hidden_dim),
                                       # nn.BatchNorm1d(args.hidden_dim),
                                       nn.Softplus(),
                                       # nn.Dropout(0.5),
                                       nn.Linear(args.hidden_dim, 2*args.z_depth*args.z_dim**2))


        # initialize weights to N(0, 0.001) and biases to 0 (cf SSL section 4.4)
        for p in self.parameters():
            p.data.normal_(0, 0.001)
            if p.ndimension() == 1: p.data.fill_(0.)

    # q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- SSL paper eq 4
    def encode_z(self, x, y, n=None):
        if n is None:
            n=x.shape[0]
        xconv = self.encoder_z_cnn(x).view(n, -1)
        xy = torch.cat([xconv, y], dim=1)
        mu, logsigma = self.encoder_z(xy).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())

    # q(y|x) = Categorical(y|pi_phi(x)) -- SSL paper eq 4
    def encode_y(self, x):
        xconv = self.encoder_y_cnn(x).view(x.shape[0], -1)
        return D.OneHotCategorical(logits=self.encoder_y(xconv))

    # p(x|y,z) = Bernoulli
    def decode(self, y, z):
        yz = torch.cat([y,z], dim=1)
        yz = self.decoder(yz).view(yz.shape[0], args.image_dims[0], args.image_dims[1], args.image_dims[2])
        yz = self.decoder_tcnn(yz).view(yz.shape[0], -1)
        return D.Bernoulli(logits=yz)

    # classification model q(y|x) using the trained q distribution
    def forward(self, x):
        y_probs = self.encode_y(x).probs
        return y_probs.max(dim=1)[1]  # return pred labels = argmax

def loss_components_fn(x, y, z, p_y, p_z, p_x_yz, q_z_xy):
    # SSL paper eq 6 for an given y (observed or enumerated from q_y)
    x = x.view(x.shape[0], -1)
    if args.z_depth == 1:
        return - p_x_yz.log_prob(x).sum(1) \
               - p_y.log_prob(y) \
               + mix_gaussian_loss_1d(z.view(args.batch_size, args.z_depth, args.z_dim, args.z_dim), p_z(z.view(args.batch_size, args.z_depth, args.z_dim, args.z_dim))).sum(1).sum(1) \
               + q_z_xy.log_prob(z).sum(1)
    else:
        return - p_x_yz.log_prob(x).sum(1) \
               - p_y.log_prob(y) \
               + mix_gaussian_loss(z.view(args.batch_size, args.z_depth, args.z_dim, args.z_dim), p_z(z.view(args.batch_size, args.z_depth, args.z_dim, args.z_dim))).sum(1).sum(1) \
               + q_z_xy.log_prob(z).sum(1)


# --------------------
# Train and eval
# --------------------

def train_epoch(model, labeled_dataloader, unlabeled_dataloader, loss_components_fn, optimizer, epoch, args):
    model.train()

    n_batches = len(labeled_dataloader) + len(unlabeled_dataloader)
    n_unlabeled_per_labeled = len(unlabeled_dataloader) // len(labeled_dataloader) + 1

    labeled_dataloader_batch = iter(labeled_dataloader)
    unlabeled_dataloader_batch = iter(unlabeled_dataloader)

    with tqdm(total=n_batches, desc='epoch {} of {}'.format(epoch+1, args.n_epochs)) as pbar:
        for i in range(n_batches):
            is_supervised = i % n_unlabeled_per_labeled == 0

            # get batch from respective dataloader
            if is_supervised:
                try:
                    x, y = next(labeled_dataloader_batch)
                except StopIteration: # til cifar10. Evt find anden løsning ???
                    labeled_dataloader_batch = iter(labeled_dataloader)
                    x, y = next(labeled_dataloader_batch)
                y = one_hot(y, args.y_dim).to(args.device)
            else:
                try:
                    x, y = next(unlabeled_dataloader_batch)
                except StopIteration: # til cifar10. Evt find anden løsning ???
                    unlabeled_dataloader_batch = iter(unlabeled_dataloader)
                    x, y = next(unlabeled_dataloader_batch)
                
                y = None

            x = x.to(args.device)
            # x = x.to(args.device).view(x.shape[0], -1)

            # compute loss -- SSL paper eq 6, 7, 9
            q_y = model.encode_y(x)
            # labeled data loss -- SSL paper eq 6 and eq 9
            if y is not None:
                q_z_xy = model.encode_z(x, y)
                z = q_z_xy.rsample()
                # z = model.encode_z(x, y)
                p_x_yz = model.decode(y, z)
                loss = loss_components_fn(x, y, z, model.p_y, model.p_z, p_x_yz, q_z_xy)
                loss -= args.alpha * args.n_labeled * q_y.log_prob(y)  # SSL eq 9
            # unlabeled data loss -- SSL paper eq 7
            else:
                # marginalize y according to q_y
                loss = - q_y.entropy()
                for y in q_y.enumerate_support():
                    q_z_xy = model.encode_z(x, y)
                    z = q_z_xy.rsample()
                    # z = model.encode_z(x, y)
                    p_x_yz = model.decode(y, z)
                    L_xy = loss_components_fn(x, y, z, model.p_y, model.p_z, p_x_yz, q_z_xy)
                    loss += q_y.log_prob(y).exp() * L_xy
            loss = loss.mean(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update trackers
            pbar.set_postfix(loss='{:.3f}'.format(loss.item()))
            pbar.update()


@torch.no_grad()
def evaluate(model, dataloader, epoch, args):
    model.eval()

    accurate_preds = 0

    with tqdm(total=len(dataloader), desc='eval') as pbar:
        for i, (x, y) in enumerate(dataloader):
            x = x.to(args.device) # .view(x.shape[0], -1)
            y = y.to(args.device)
            preds = model(x)

            accurate_preds += (preds == y).sum().item()

            pbar.set_postfix(accuracy='{:.4f}'.format(accurate_preds / ((i+1) * args.batch_size)))
            pbar.update()



    output = (epoch != None)*'Epoch {} -- '.format(epoch) + 'Test set accuracy: {:.4f}'.format(accurate_preds / (args.batch_size * len(dataloader)))
    print(output)
    print(output, file=open(args.results_file, 'a'))
    generate(model, test_dataloader.dataset, args, epoch)
    if args.calc_nll:
        print("Calculating variational lower bound...")
        print("NLL = ", calc_nll(model, test_dataloader.dataset, 100))

def train_and_evaluate(model, labeled_dataloader, unlabeled_dataloader, test_dataloader, loss_components_fn, optimizer, args):
    for epoch in range(args.n_epochs):
        train_epoch(model, labeled_dataloader, unlabeled_dataloader, loss_components_fn, optimizer, epoch, args)
        evaluate(model, test_dataloader, epoch, args)

        # save weights
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'ssvae_model_state_hdim{}_zdim{}.pt'.format(
                                                                        args.hidden_dim, args.z_dim)))

        # show samples -- SSL paper Figure 1-b

        # if epoch % 10 == 0:
        generate(model, test_dataloader.dataset, args, epoch)


# --------------------
# Visualize
# --------------------

@torch.no_grad()
def generate(model, dataset, args, epoch=None, n_samples=10):
    n_samples_per_label = 10

    # some interesting samples per paper implementation
    idxs = [7910, 8150, 3623, 2645, 4066, 9660, 5083, 948, 2595, 2]

    x = torch.stack([dataset[i][0] for i in idxs], dim=0).to(args.device)
    y = torch.stack([torch.tensor(dataset[i][1]) for i in idxs], dim=0).to(args.device)
    y = one_hot(y, args.y_dim)

    q_z_xy = model.encode_z(x.view(n_samples_per_label, args.image_dims[0], args.image_dims[1], args.image_dims[2]), y)
    z = q_z_xy.loc
    z = z.repeat(args.y_dim, 1, 1).transpose(0, 1).contiguous().view(-1, args.z_depth*args.z_dim**2) # args.z_depth, args.z_dim, args.z_dim)

    # hold z constant and vary y:
    y = torch.eye(args.y_dim).repeat(n_samples_per_label, 1).to(args.device)
    generated_x = model.decode(y, z).probs.view(n_samples_per_label, args.y_dim, *args.image_dims)
    generated_x = generated_x.contiguous().view(-1, *args.image_dims)  # out (n_samples * n_label, C, H, W)

    x = make_grid(x.cpu(), nrow=1)
    spacer = torch.ones(x.shape[0], x.shape[1], 5)
    generated_x = make_grid(generated_x.cpu(), nrow=args.y_dim)
    image = torch.cat([x, spacer, generated_x], dim=-1)
    save_image(image,
               os.path.join(args.output_dir, 'analogies_sample' + (epoch != None)*'_at_epoch_{}'.format(epoch) + '.png'),
               nrow=args.y_dim)

    #if args.dataset=='cifar10' and generate_for_cifar10:
    #if args.generate_samples:
    n=10
    digit_size = args.image_dims[1]
    figure = np.zeros((digit_size * n, digit_size * n))
    for j in range(n):
        for i in range(n):
            y_sample = model.p_y.sample()    # torch.nn.functional.one_hot(torch.LongTensor([i]).cuda(), num_classes=10)
            y_sample = y_sample.float()
            z_sample = sample_z(model, 1)  # np.random.normal(size=(1,args.z_dim))
            z_sample = torch.tensor(z_sample) #torch.FloatTensor(z_sample)#.cuda()
            x_decoded= model.decode(y_sample, z_sample.view(1, -1)).probs.view(*args.image_dims)
            #x_decoded= unlabeled_vae.decoder(z_sample,y_sample,y_output=y_sample)
            #x_decoded = generator.predict([y_sample, z_sample], batch_size=1)
            digit = x_decoded
            d_x = i * digit_size
            d_y = j * digit_size
            figure[d_x:d_x + digit_size, d_y:d_y + digit_size] = digit.cpu().detach()
    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.savefig(os.path.join(args.output_dir, 'generated_samples' + (epoch != None)*'_at_epoch_{}'.format(epoch) + '.png'))
    # plt.close()
    save_image(torch.tensor(figure),
            os.path.join(args.output_dir, 'true_samples' + (epoch != None)*'_at_epoch_{}'.format(epoch) + '.png'),
            nrow=n)

@torch.no_grad()
def vis_styles(model, args):
    assert args.z_dim == 2, 'Style viualization requires z_dim=2'

    for y in range(2,5):
        y = one_hot(torch.tensor(y).unsqueeze(-1), args.y_dim).expand(100, args.y_dim).to(args.device)

        # order the first dim of the z latent
        c = torch.linspace(-5, 5
                           , 10).view(-1,1).repeat(1,10).reshape(-1,1)
        z = torch.cat([c, torch.zeros_like(c)], dim=1).reshape(100, 2).to(args.device)

        # combine into z and pass through decoder
        x = model.decode(y, z).probs.view(y.shape[0], *args.image_dims)
        save_image(x.cpu(),
                   os.path.join(args.output_dir, 'latent_var_grid_sample_c1_y{}.png'.format(y[0].nonzero().item())),
                   nrow=10)

        # order second dim of latent and pass through decoder
        z = z.flip(1)
        x = model.decode(y, z).probs.view(y.shape[0], *args.image_dims)
        save_image(x.cpu(),
                   os.path.join(args.output_dir, 'latent_var_grid_sample_c2_y{}.png'.format(y[0].nonzero().item())),
                   nrow=10)

def sample_z(model, sample_batch_size):
    model.train(False)
    data = torch.zeros(sample_batch_size, args.z_depth, args.z_dim, args.z_dim)
    data = data.cuda()
    for i in range(args.z_dim):
        for j in range(args.z_dim):
            data_v = Variable(data, volatile=True)
            out  = model.p_z.forward(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

def calc_nll(model, test_data, num_samples):
    """ Calculates variational lower bound through Monte Carlo sampling """
    nll = 0
    for x in test_data:
        x_img = x[0].view(1, args.image_dims[0], args.image_dims[1], args.image_dims[2]).cuda()
        y = torch.tensor([0,0,0,0,0,0,0,0,0,0])
        y[x[1]]=1
        y = y.float().cuda().view(1, -1)
        # q_y = model.encode_y(x)
        q_z_xy = model.encode_z(x_img, y)
        x_flat = x_img.view(1, -1)
        loss = 0
        for _ in range(num_samples):
            z = q_z_xy.rsample()
            p_x_yz = model.decode(y, z)
            if args.z_depth == 1:
                loss += - p_x_yz.log_prob(x_flat).sum(1) \
                       - model.p_y.log_prob(y) \
                       + mix_gaussian_loss_1d(z.view(1, args.z_depth, args.z_dim, args.z_dim), model.p_z(z.view(1, args.z_depth, args.z_dim, args.z_dim))).sum(1).sum(1) \
                       + q_z_xy.log_prob(z).sum(1)
            else:
                loss += - p_x_yz.log_prob(x_flat).sum(1) \
                       - model.p_y.log_prob(y) \
                       + mix_gaussian_loss(z.view(1, args.z_depth, args.z_dim, args.z_dim), model.p_z(z.view(1, args.z_depth, args.z_dim, args.z_dim))).sum(1).sum(1) \
                       + q_z_xy.log_prob(z).sum(1)
            # loss += loss_components_fn(x_flat, y, z, model.p_y, model.p_z, p_x_yz, q_z_xy)
        nll += loss/num_samples
    nll /= len(test_data)
    return nll

# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    args.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda != None else 'cpu')
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)


    assert args.dataset == 'mnist' or args.dataset == 'cifar10'
    if args.dataset=='cifar10':
        args.image_dims=(3,32,32)
        sample_op = lambda x : sample_from_mix_gaussian(x, args.mixture_comps) # last arg is nr_gaussian_mix
    elif args.dataset=='mnist':
        args.image_dims=(1,28,28)
        sample_op = lambda x : sample_from_mix_gaussian_1d(x, args.mixture_comps) # last arg is nr_gaussian_mix

    # dataloaders
    labeled_dataloader, unlabeled_dataloader, test_dataloader = fetch_dataloaders(args)

    # model
    model = SSVAE(args).to(args.device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.restore_file:
        # load model and optimizer states
        state = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(state)
        # set up paths
        args.output_dir = os.path.dirname(args.restore_file)
    args.results_file = os.path.join(args.output_dir, args.results_file)

    print('Loaded settings and model:')
    print(pprint.pformat(args.__dict__))
    print(model)
    print(pprint.pformat(args.__dict__), file=open(args.results_file, 'a'))
    print(model, file=open(args.results_file, 'a'))

    if args.train:
        train_and_evaluate(model, labeled_dataloader, unlabeled_dataloader, test_dataloader, loss_components_fn, optimizer, args)

    if args.evaluate:
        evaluate(model, test_dataloader, None, args)

    if args.generate:
        generate(model, test_dataloader.dataset, args)

    if args.vis_styles:
        vis_styles(model, args)



