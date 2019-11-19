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

from made import MADE
from constants import *


# ------------------------------------------------------------------------------
def run_epoch(split, upto=None, save_loc=""):
    torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    nsamples = 1 if split == 'train' else args.samples
    x = xtr if split == 'train' else xte
    N,D = x.size()
    B = 100 # batch size
    nsteps = N//B if upto is None else min(N//B, upto)
    lossfs = []
    for step in range(nsteps):
        
        # fetch the next batch of data
        xb = Variable(x[step*B:step*B+B])
        
        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = torch.zeros_like(xb)
        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            model.update_masks(resample_hidden_masks=False)
            if step % args.resample_every == 0 or split == 'test': # if in test, cycle masks every time
                model.update_masks(resample_hidden_masks=True)
            # forward the model
            xbhat += model(xb)
        xbhat /= nsamples
        
        # evaluate the binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(xbhat, xb, size_average=False) / B
        lossf = loss.data.item()
        lossfs.append(lossf)
        
        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

    if save_loc != '':
        torch.save(model, save_loc)
        print('Model parameters saved to ' + str(save_loc))
        
    # print("%s epoch average loss: %f" % (split, np.mean(lossfs)))
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--hiddens', type=str, default='500', help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
    parser.add_argument('-n', '--num-masks', type=int, default=1, help="Number of orderings for order/connection-agnostic training")
    parser.add_argument('-r', '--resample-every', type=int, default=20, help="For efficiency we can choose to resample orders/masks only once every this many steps")
    parser.add_argument('-s', '--samples', type=int, default=1, help="How many samples of connectivity/masks to average logits over during inference")
    parser.add_argument('-m', '--model-name', required=True, type=str, help="Name of model for save location")
    args = parser.parse_args()
    # --------------------------------------------------------------------------
    
    save_path = Path('./checkpoints') / ('made_' + args.model_name)
    assert not save_path.is_dir()
    save_path.mkdir()

    # reproducibility is good
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # load the dataset
    print("loading binarized mnist from", DATA_PATH)
    mnist = np.load(DATA_PATH)
    xtr, xte = mnist['train_data'], mnist['valid_data']
    xtr = torch.from_numpy(xtr).cuda() # [50000, 784]
    xte = torch.from_numpy(xte).cuda() # [10000, 784]

    # construct model and ship to GPU
    hidden_list = list(map(int, args.hiddens.split(',')))
    model = MADE(xtr.size(1), hidden_list, xtr.size(1), num_masks=args.num_masks, orderings=MNIST_ORDERINGS) # TODO: natural_ordering
    print("number of model parameters:",sum([np.prod(p.size()) for p in model.parameters()]))
    model.cuda()

    # set up the optimizer
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)
    
    # start the training
    for epoch in tqdm(range(100)):
        # print("epoch %d" % (epoch, ))
        run_epoch('test', upto=5) # run only a few batches for approximate test accuracy
        scheduler.step(epoch)
        save_loc = save_path / f'{epoch:03d}_params.pt' if (epoch + 1) % 20 == 0 else ''
        run_epoch('train', save_loc=save_loc)
    
    print("optimization done. full test set eval:")
    run_epoch('test')

