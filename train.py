"""
Trains MADE on Binarized MNIST, which can be downloaded here:
https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pathlib import Path
from tqdm import tqdm
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt

from made import MADE
from inference import Inference
from config import *
from util import *


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------

def run(c=None):
    if c is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', required=True, type=str, help="Hyperparameter config")
        args = parser.parse_args()
        hyp = set_config(args.config)
    else:
        hyp = c

    save_path = Path('./checkpoints') / ('made_' + hyp['MODEL_NAME'])
    if save_path.is_dir():
        print('Skipping; save directory already exists.')
        return False
    save_path.mkdir()

    # reproducibility is good
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # load the dataset (TODO: refactor to use dataloaders)
    if hyp['DATASET'] == 'mnist':
        print("Loading binarized mnist from", DATA_PATH)
        mnist = np.load(DATA_PATH)
        xtr, xte = mnist['train_data'], mnist['valid_data']
        xtr = torch.from_numpy(xtr).cuda() # [50000, 784]
        xte = torch.from_numpy(xte).cuda() # [10000, 784]
        # train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
        #     transform = transforms.Compose([transforms.ToTensor()]))
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=100,
        #                                           shuffle=True, num_workers=2)
        # test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
        #     transform = transforms.Compose([transforms.ToTensor()]))
        # test_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
        #                                          shuffle=False, num_workers=2)
        orderings = hyp['MNIST_ORDERINGS']

    elif hyp['DATASET'] == 'cifar10':
        raise NotImplementedError
        # train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
        #     transform = transforms.Compose([transforms.ToTensor()]))
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=16,
        #                                           shuffle=True, num_workers=2)
        # test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
        #     transform = transforms.Compose([transforms.ToTensor()]))
        # test_loader = torch.utils.data.DataLoader(test_set, batch_size=16,
        #                                          shuffle=False, num_workers=2)
        # orderings = hyp['CIFAR10_ORDERINGS']

    # construct model and ship to GPU
    # x_sample, _ = iter(train_loader).next()
    # flat_length = x_sample.size(1) * x_sample.size(2) * x_sample.size(3)
    flat_length = xtr.size(1)
    model = MADE(flat_length, hyp['HIDDEN_LIST'], flat_length, num_masks=hyp['NUM_MASKS'], orderings=orderings)
    print("number of model parameters:",sum([np.prod(p.size()) for p in model.parameters()]))
    model.cuda()

    # set up the optimizer
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)
    losses = {'train': [], 'test': []}

    def run_epoch(split, upto=None, save_loc=""):
        torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches
        model.train() if split == 'train' else model.eval()
        nsamples = len(model.orderings) if hyp['RUN_EVERY_ORDERING'] else 1
        lossfs = []
        kls = []
        bces = []

        # commented out with removal of dataloader approach
        # step = 0
        # loader = train_loader if split == 'train' else test_loader
        # for xb, _ in loader:

        x = xtr if split == 'train' else xte
        N,D = x.size()
        B = 100 # batch size
        nsteps = N//B if upto is None else min(N//B, upto)
        for step in range(nsteps):

            # commented out with removal of dataloader approach
            # B = xb.size(0)
            # xb = xb.cuda()
            # xb = xb.permute(0, 2, 3, 1)
            # xb = xb.reshape(B, -1)

            
            # fetch the next batch of data
            xb = Variable(x[step*B:step*B+B])
            
            # get the logits, potentially run the same batch a number of times, resampling each time
            xbhat = torch.zeros_like(xb)
            outs = []
            for s in range(nsamples):
                # perform order/connectivity-agnostic training by resampling the masks
                model.update_ordering()
                # forward the model
                out = model(xb)
                outs.append(out)
                xbhat += out
            xbhat /= nsamples

            # evaluate the binary cross entropy loss
            bce_loss = F.binary_cross_entropy_with_logits(xbhat, xb, size_average=False) / B # TODO: this has to be changed for CIFAR

            # evaluate the kl loss
            kl_loss = 0
            loss = None
            if hyp['KL_LOSS']:
                for out in outs:
                    kl_loss = kl_loss + F.kl_div(F.sigmoid(out), F.sigmoid(xbhat))
                loss = kl_loss + bce_loss
                kls.append(kl_loss.data.item())
            else:
                loss = bce_loss

            bces.append(bce_loss.data.item())
            lossf = loss.data.item()
            lossfs.append(lossf)
            
            # backward/update
            if split == 'train':
                opt.zero_grad()
                loss.backward()
                opt.step()

            if step % hyp['RESAMPLE_EVERY'] == 0 or split == 'test': # if in test, cycle masks every time
                model.update_masks()

            # step += 1
            # if upto is not None and step == upto:
            #     break

        if save_loc != '':
            torch.save(model, save_loc)
            print('Model parameters saved to ' + str(save_loc))

        print("%s epoch average losses (total, KL, bce): %f, %f, %f" % (split, np.mean(lossfs), np.mean(kls) if hyp['KL_LOSS'] else 0.0, np.mean(bces)))

        losses[split].append((np.mean(lossfs), np.mean(kls) if hyp['KL_LOSS'] else 0.0, np.mean(bces)))

    # start the training
    for epoch in tqdm(range(100)):
        # print("epoch %d" % (epoch, ))
        if epoch > 0:
            run_epoch('test', upto=5) # run only a few batches for approximate test accuracy
        scheduler.step(epoch)
        save_loc = save_path / f'{epoch:03d}_params.pt' if (epoch + 1) % 20 == 0 else ''
        run_epoch('train', save_loc=save_loc)
    
    print("optimization done. full test set eval:")
    run_epoch('test')

    save_dict(losses, save_path / 'losses.pkl')
    return True



def grid_search():
    H = [[500], [500,500], [500,500,500], [1000], [1000,1000]]
    N = [1, 5, 17]
    R = [True, False]
    O = [1, 8]
    for h in range(len(H)):
        for n in range(len(N)):
            for r in range(len(R)):
                for o in range(len(O)):
                    c = {
                        'HIDDEN_LIST': H[h],
                        'NUM_MASKS': N[n],
                        'RESAMPLE_EVERY': 20,
                        'RUN_EVERY_ORDERING': R[r],
                        'MODEL_NAME': f'mnist_gridsearch_{h}{n}{r}{o}',
                        'DATASET': 'mnist',
                        'MNIST_ORDERINGS': MNIST_ORDERINGS[:O[o]],
                        'KL_LOSS': False,
                    }
                    print(f'RUNNING CASE {h}{n}{r}{o}')
                    f = run(c)
                    if f:
                        for i in range(5):
                            I = Inference('made_' + c['MODEL_NAME'], f'0{20*i+19}_params.pt')
                            x = I.sample_single_ordering(16)
                            vis = x.view(4, 28*4, 28).transpose(0,1).reshape(4*28, 4*28)
                            plt.imshow(vis.cpu().numpy(), cmap='gray')
                            plt.savefig(Path('./checkpoints') / ('made_' + c['MODEL_NAME']) / f'0{20*i+19}.png')


if __name__ == '__main__':
    # grid_search()
    c = {
        'HIDDEN_LIST': [500,500],
        'NUM_MASKS': 3,
        'RESAMPLE_EVERY': 1,
        'RUN_EVERY_ORDERING': False,
        'MODEL_NAME': 'mnist_55_3m_1o',
        'DATASET': 'mnist',
        'MNIST_ORDERINGS': MNIST_ORDERINGS[:1], # [MNIST_ORDERINGS[0], MNIST_ORDERINGS[2]],
        'KL_LOSS': False,
    }
    run(c)