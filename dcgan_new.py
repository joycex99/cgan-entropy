""" Conditional DCGAN for MNIST images generations.
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from data_loader import get_loader

# SAMPLE_SIZE = 80
NUM_LABELS = 8

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        # TODO: Fix this with GPU and cuda 
        self.ngpu = 0 
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d( nz, ngf , 4, 1, 0, bias=False),
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        # TODO: Fix this with GPU and cuda 
        self.ngpu = 0 
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_generator(netG, netD, num_epochs, dataloader, device, batch_size, nz):

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    import pdb; pdb.set_trace()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            # print(real_cpu.size())
            # print("batch size" + str(b_size))
            label = torch.full((batch_size,), real_label, device=device)
            # Forward pass real batch through D
            # print(label.size())
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            # reshape to batch size 
            # output = output.reshape((batch_size, int(output.size()[0]/batch_size)))
            # print(output.size())
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
            iters += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DCGAN')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default=64)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default=0.01)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--nz', type=int, default=100,
                        help='Number of dimensions for input noise.')
    parser.add_argument('--beta1', type = float, default = 0.5, 
                        help="Beta1 hyperparam for Adam optimizers.")
    parser.add_argument('--cuda', action='store_true',
                        help='Enable cuda')
    parser.add_argument('--save_every', type=int, default=1,
                        help='After how many epochs to save the model.')
    parser.add_argument('--print_every', type=int, default=50,
            help='After how many epochs to print loss and save output samples.')
    parser.add_argument('--save_dir', type=str, default='models',
            help='Path to save the trained models.')
    parser.add_argument('--image_size',type=int, default=64)
    parser.add_argument('--crop_size',type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--samples_dir', type=str, default='samples',
            help='Path to save the output samples.')
    # parser.add_argument('--emotion_dir', type=str, default='/Users/evazhang/Downloads/entropy-gan-master/data/Emotion', help='emotion data directory.')
    # parser.add_argument('--image_dir', type=str, default='/Users/evazhang/Downloads/entropy-gan-master/data/data/ck_align', help='image data directory')
    parser.add_argument('--emotion_dir', type=str, default='/Users/joycexu/Documents/cs236/entropy-gan/data/Emotion', help='emotion data directory.')
    parser.add_argument('--image_dir', type=str, default='/Users/joycexu/Documents/cs236/entropy-gan/data/data/ck_align', help='image data directory')
    
    parser.add_argument('--cls', type=int, default=7)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--ithfold', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', help='train|valid')
    parser.add_argument('--nc', type=int, default = 3, help = 'nchannels, default rgb = 3')
    parser.add_argument('--ndf', type = int, default = 64, help = 'size of feature map in discriminator')
    parser.add_argument('--ngf', type = int, default = 64, help = 'size of feature map in generator')

    args = parser.parse_args()
   
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.samples_dir):
        os.mkdir(args.samples_dir)

    if os.path.exists(args.emotion_dir):
        print(os.path.isdir(args.emotion_dir + '/S010'))

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # initialize train loaders for ck+ data 
    train_loader, valid_loader, _ = get_loader(args)

    # initialize models 
    netD = Discriminator(args.ndf, args.nc)
    netD.apply(weights_init)
    netG = Generator(args.nz, args.ngf, args.nc)
    netG.apply(weights_init)
    train_generator(netG, netD, args.epochs, train_loader, device, args.batch_size, args.nz)





