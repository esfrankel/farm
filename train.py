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
from config import *


# ------------------------------------------------------------------------------
def run_epoch(split, upto=None, save_loc=""):
    torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    nsamples = args.samples
    model.nsamples = nsamples
    x = xtr if split == 'train' else xte
    N,D = x.size()
    B = 100 # batch size
    nsteps = N//B if upto is None else min(N//B, upto)
    lossfs = []
    kls = []
    for step in range(nsteps):
        
        # fetch the next batch of data
        xb = Variable(x[step*B:step*B+B])
        
        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = torch.zeros_like(xb)
        total_w = torch.zeros_like(xb)
        outs = []
        weights = []
        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            model.update_masks(resample_hidden_masks=False)
            # forward the model
            w = torch.from_numpy(model.m[-1] + 1).cuda() # the ordering
            out = model(xb)
            outs.append(out)
            weights.append(w)
            xbhat += out * w
            total_w += w

        xbhat /= total_w
        
        # evaluate the binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(xbhat, xb, size_average=False) / B

        if args.use_kl:
            kl_loss = 0
            for i in range(nsamples):
                kl_loss += (F.kl_div(F.logsigmoid(outs[i]), F.sigmoid(xbhat), reduction='none') * weights[i])
                # import pdb; pdb.set_trace()
            kl_loss /= total_w
            kl_loss = kl_loss.mean()
            loss += kl_loss * 100 # tune the loss weight factor
            kls.append(kl_loss.data.item())

        lossf = loss.data.item()
        lossfs.append(lossf)
        
        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        if step % args.resample_every == 0 or split == 'test': # if in test, cycle masks every time
            model.update_masks(resample_hidden_masks=True)

    if save_loc != '':
        torch.save(model, save_loc)
        print('Model parameters saved to ' + str(save_loc))
        
    print("%s epoch average loss: %f (total); %f (KL)" % (split, np.mean(lossfs), np.mean(kls) if len(kls) > 0 else 0.0))
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--hiddens', type=str, default='500', help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
    parser.add_argument('-n', '--num-masks', type=int, default=1, help="Number of orderings for order/connection-agnostic training")
    parser.add_argument('-r', '--resample-every', type=int, default=20, help="For efficiency we can choose to resample orders/masks only once every this many steps")
    parser.add_argument('-s', '--samples', type=int, default=1, help="How many samples of connectivity/masks to average logits over during inference")
    parser.add_argument('-m', '--model-name', required=True, type=str, help="Name of model for save location")
    parser.add_argument('-o', '--num-orderings', type=int, default=1, help="Number of orderings to use")
    parser.add_argument('-k', '--use-kl', default=False, action='store_true', help="Whether to use KL loss")
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
    o = {
        1: [MNIST_ORDERINGS[i] for i in [0]],
        2: [MNIST_ORDERINGS[i] for i in [0, 2]],
        4: [MNIST_ORDERINGS[i] for i in [0, 2, 5, 7]],
        8: [MNIST_ORDERINGS[i] for i in range(8)]
    }
    hidden_list = list(map(int, args.hiddens.split(',')))
    model = MADE(xtr.size(1), hidden_list, xtr.size(1), num_masks=args.num_masks, orderings=o[args.num_orderings])
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
