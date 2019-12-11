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
model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(lr, nr_resnet, nr_filters)
torch.cuda.set_device(2)
device_1 = torch.device("cuda:2")
device_2 = torch.device("cuda:3")

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
    
    loss_op   = lambda real, fake, device : discretized_mix_logistic_loss_1d(real, fake, device)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, nr_logistic_mix)
elif 'cifar' in dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, train=False, 
                    transform=ds_transforms), batch_size=batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake, device : discretized_mix_logistic_loss(real, fake, device)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(dataset))
    
model_1 = PixelCNN(nr_resnet=nr_resnet, nr_filters=nr_filters, 
            input_channels=input_channels, nr_logistic_mix=nr_logistic_mix) # Note: Using Sideways
model_1 = model_1.cuda()
if load_params:
    load_part_of_model(model_1, load_params)
    # model.load_state_dict(torch.load(load_params))
    print('model parameters loaded')
    
optimizer_1 = optim.Adam(model_1.parameters(), lr=lr)
scheduler_1 = lr_scheduler.StepLR(optimizer_1, step_size=1, gamma=lr_decay)

model_2 = PixelCNNSideways(nr_resnet=nr_resnet, nr_filters=nr_filters, 
            input_channels=input_channels, nr_logistic_mix=nr_logistic_mix) # Note: Using Sideways
model_2 = model_2.cuda()
model_2.to(device_2)
if load_params:
    load_part_of_model(model_2, load_params)
    # model.load_state_dict(torch.load(load_params))
    print('model parameters loaded')
    
optimizer_2 = optim.Adam(model_2.parameters(), lr=lr)
scheduler_2 = lr_scheduler.StepLR(optimizer_2, step_size=1, gamma=lr_decay)

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for j in range(obs[2]): # Note: Sampling is flipped here to fill in js first
        for i in tqdm(range(obs[1])):     
            data_v = Variable(data, volatile=True)
            out   = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
writes = 0
for epoch in range(max_epochs):
    model_1.train(True)
    model_2.train(True)
    torch.cuda.synchronize()
    train_loss_1, train_loss_2, train_JSD = 0., 0., 0.
    time_ = time.time()
    model_1.train()
    model_2.train()
    
    for batch_idx, (input,_) in enumerate(tqdm(train_loader)):
        input = input.cuda()
        input_var_1 = Variable(input)
        input_var_2 = Variable(input.to(device_2))
        output_1 = model_1(input_var_1)
        output_2 = model_2(input_var_2)
        loss_1, loss_2, JSD = discretized_mix_logistic_loss_joint(input_var_1, output_1, output_2.to(device_1), device_1)
        loss_2 = loss_2.to(device_2)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        loss_1.backward(retain_graph=True)
        loss_2.backward()
        optimizer_1.step()
        optimizer_2.step()
        train_loss_1 += loss_1.data.item()
        train_loss_2 += loss_2.data.item()
        train_JSD += JSD.data.item()
        if (batch_idx +1) % print_every == 0 : 
            deno = print_every * batch_size * np.prod(obs) * np.log(2.)
            writer.add_scalar('train/bpd', (train_loss_1 / deno), writes)
            print('loss 1 : {:.4f}, time : {:.4f}'.format(
                (train_loss_1 / deno), 
                (time.time() - time_)))
            print('loss 2 : {:.4f}, time : {:.4f}'.format(
                (train_loss_2 / deno), 
                (time.time() - time_)))
            print('loss JSD : {:.4f}, time : {:.4f}'.format(
                (JSD / deno), 
                (time.time() - time_)))
            train_loss_1, train_loss_2 = 0., 0.
            writes += 1
            time_ = time.time()
            
    # decrease learning rate
    scheduler_1.step()
    scheduler_2.step()
    
    torch.cuda.synchronize()
    model_1.eval()
    model_2.eval()
    test_loss_1, test_loss_2, test_JSD = 0., 0., 0.
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.cuda()
        input_var_1 = Variable(input)
        input_var_2 = Variable(input.to(device_2))
        output_1 = model_1(input_var_1)
        output_2 = model_2(input_var_2)
        loss_1, loss_2, JSD = discretized_mix_logistic_loss_joint(input_var_1, output_1, output_2.to(device_1), device_1)
        loss_2 = loss_2.to(device_2)
        # loss_1 = loss_op(input_var, output_1, device_1) #TODO: FINISH 2 MODEL VERSION
        # loss_2 = loss_op(input_var_2, output_2, device_2) #TODO: FINISH 2 MODEL VERSION
        test_loss_1 += loss_1.data.item()
        test_loss_2 += loss_2.data.item()
        test_JSD += JSD.data.item()
        del input, input_var_1, input_var_2, loss_1, loss_2, output_1, output_2, JSD
    deno = batch_idx * batch_size * np.prod(obs) * np.log(2.)
    writer.add_scalar('test/bpd', (test_loss_1 / deno), writes)
    print('test loss 1: %s' % (test_loss_1 / deno))
    print('test loss 2: %s' % (test_loss_2 / deno))
    print('test loss JSD: %s' % (test_JSD / deno))
    
    if (epoch + 1) % save_interval == 0: 
        torch.save(model_1.state_dict(), 'models/model_1_{}_{}.pth'.format(model_name, epoch))
        torch.save(model_2.state_dict(), 'models/model_2_{}_{}.pth'.format(model_name, epoch))
        # print('sampling...')
        # sample_t_1 = sample(model_1)
        # sample_t_1 = rescaling_inv(sample_t_1)
        # sample_t_2 = sample(model_2)
        # sample_t_2 = rescaling_inv(sample_t_2)
        # utils.save_image(sample_t,'images/model_1_{}_{}.png'.format(model_name, epoch), nrow=5, padding=0)
        # utils.save_image(sample_t_2,'images/model_2_{}_{}.png'.format(model_name, epoch), nrow=5, padding=0)
