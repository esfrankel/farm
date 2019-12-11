import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from utils import * 
from model import * 
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import save_image

# data I/O
data_dir = 'data' # location for dataset
save_dir = 'models' # location for parameter checkpoints and samples
dataset = 'cifar' # cifar or mnist
print_every = 50 # how many iter between print statements
save_interval = 5 # how many epoch to write checkpoint/samples
# load_params = './runs/pretrained/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth' # restore training from previous model checkpoint, default is None
load_params = None

# model
nr_resnet = 5 # num of resid blocks per stage
nr_filters = 160 # num of filters 
nr_logistic_mix = 10 # num of logistic components in mixture
lr = 0.0002 # base learning rate
lr_decay = 0.999995 # lr decay after every step in optimization
batch_size = 32 # batch size, default is 64
max_epochs = 5000 # max num of epochs
seed = 1 # random seed

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)
model_name = 'baseline_pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(lr, nr_resnet, nr_filters)
torch.cuda.set_device(0)

# assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)

writer = SummaryWriter(log_dir=os.path.join('runs', model_name))
sample_batch_size = 25
obs = (1, 28, 28) if 'mnist' in dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
if 'mnist' in dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(data_dir, train=False, 
                    transform=ds_transforms), batch_size=batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, nr_logistic_mix)
elif 'cifar' in dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, train=False, 
                    transform=ds_transforms), batch_size=batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(dataset))
    
model = PixelCNN(nr_resnet=nr_resnet, nr_filters=nr_filters, 
            input_channels=input_channels, nr_logistic_mix=nr_logistic_mix)
model = model.cuda()
if load_params:
    load_part_of_model(model, load_params)
    # model.load_state_dict(torch.load(load_params))
    print('model parameters loaded')
    
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in tqdm(range(obs[1])):
        for j in range(obs[2]):
            data_v = Variable(data, volatile=True)
            out   = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
writes = 0
for epoch in range(max_epochs):
    model.train(True)
    torch.cuda.synchronize()
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(tqdm(train_loader)):
        input = input.cuda()
        input = Variable(input)
        output = model(input)
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        if (batch_idx +1) % print_every == 0 : 
            deno = print_every * batch_size * np.prod(obs) * np.log(2.)
            writer.add_scalar('train/bpd', (train_loss / deno), writes)
            print('loss : {:.4f}, time : {:.4f}'.format(
                (train_loss / deno), 
                (time.time() - time_)))
            train_loss = 0.
            writes += 1
            time_ = time.time()
            
    # decrease learning rate
    scheduler.step()
    
    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.cuda()
        input_var = Variable(input)
        output = model(input_var)
        loss = loss_op(input_var, output)
        test_loss += loss.data.item()
        del loss, output
    deno = batch_idx * batch_size * np.prod(obs) * np.log(2.)
    writer.add_scalar('test/bpd', (test_loss / deno), writes)
    print('test loss : %s' % (test_loss / deno))
    
    if (epoch + 1) % save_interval == 0: 
        torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
        # print('sampling...')
        # sample_t = sample(model)
        # sample_t = rescaling_inv(sample_t)
        # utils.save_image(sample_t,'images/{}_{}.png'.format(model_name, epoch), nrow=5, padding=0)
