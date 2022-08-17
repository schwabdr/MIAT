from __future__ import print_function
import os
import argparse
import torch
# import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

import torchvision.models as models
import torchvision.datasets as datasets

#from models.wideresnet import WideResNet
#from models.resnet import ResNet18
from utils.standard_loss import standard_loss
import torch.backends.cudnn as cudnn
from torch.autograd import Variable #deprecated??
from data import data_dataset
import numpy as np
import random

import util
import config

import projected_gradient_descent as pgd

from torchinfo import summary

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

args = config.configuration().getArgs()

resnet = models.resnet18(pretrained=False)


print(64*'=')
util.print_cuda_info()
print(64*'=')

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr

    if epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        # calculate robust loss
        # this was the original line
        #loss = standard_loss(model=model, x_natural=data, y=target, optimizer=optimizer, step_size=args.step_size,
                             #epsilon=args.epsilon, perturb_steps=args.num_steps, distance='l_inf')

        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))
            '''
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
            '''


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def craft_adversarial_example(model, x_natural, y, step_size=0.003,
                epsilon=0.031, perturb_steps=10, distance='l_inf'):

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            #need to add code to 'break' early once the example is adverial
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss_kl = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        batch_size = len(x_natural)
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.cross_entropy(model(adv), y)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    #with torch.no_grad():
    for data, target in test_loader:
        #print("data: ", data)
        #print("target: ", target)
        data, target = data.to(device), target.to(device)
        #print("data_adv", data_adv)
        
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_test_w_adv(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct_adv = 0
    eps = 1.25
    eps_iter = .05
    nb_iter = 50
    print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
    #with torch.no_grad():
    for data, target in test_loader:
        #print("data: ", data)
        #print("target: ", target)
        #print("min: ", torch.min(data)) #I can't figure why the min = -2.4291 and max = 2.7537
        #print("max: ", torch.max(data))
        x = data.detach().cpu().numpy().transpose(0,2,3,1)
        data, target = data.to(device), target.to(device)
        #epsilon was 8/255
        #perturb_steps was 40
        #data_adv = craft_adversarial_example(model=model, x_natural=data, y=target,
        #                                     step_size=0.007, epsilon=25/255,
        #                                     perturb_steps=100, distance='l_inf')
        #eps_iter is the steps size I think
        #eps is the clamp on the max change in l_inf norm?
        #data_adv = pgd.projected_gradient_descent(model, data, eps=.1, eps_iter=.007, nb_iter=100, norm=np.inf,y=None, targeted=False)
        #the following line parameters reach a 0% robust accuracy.
        #data_adv = pgd.projected_gradient_descent(model, data, eps=5.2, eps_iter=.25, nb_iter=500, norm=np.inf,y=target, targeted=False)
        #changed to targeted seems to be the key to get 0% acc.
        #eps=1.5 was 0%
        

        data_adv = pgd.projected_gradient_descent(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=target, targeted=False)
        #print("data_adv", data_adv)
        x_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1)
        output = model(data)
        output_adv = model(data_adv)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        pred_adv = output_adv.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
        

        #randomly show a grid of the adversarial examples:
        if False: #False to block out this section of code for now.
            #goal here is to randomly display an image and it's adverarial example
            stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # mean 
            x = np.clip(((x * stats[1]) + stats[0]),0,1.)
            x_adv = np.clip(((x_adv * stats[1]) + stats[0]),0,1.)
            rows = 5
            cols = 10
            fig, axes1 = plt.subplots(rows,cols,figsize=(5,5))
            lst = list(range(0, len(x)))
            random.shuffle(lst)
            #print("min/max of numpy arrays: ", np.min(x), np.max(x)) #this was very close to 0 and 1
            for j in range(5):
                for k in range(0,10,2):
                    #get a random index
                    i = lst.pop()
                    
                    axes1[j][k].set_axis_off()
                    axes1[j][k+1].set_axis_off()
                    axes1[j][k].imshow(x[i],interpolation='nearest')
                    axes1[j][k].text(0,0,classes[target[i]]) # this gets the point accross but needs fixing.
                    axes1[j][k+1].imshow(x_adv[i], interpolation='nearest')
                    axes1[j][k+1].text(0,0,classes[pred_adv[i]])
            plt.show()    
            
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Robust Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
        100. * correct_adv / len(test_loader.dataset)))

    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy



def main():
    # settings
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    #https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min/notebook
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # setup data loader
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=True)
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    trainset = data_dataset(img_path=args.nat_img_train, clean_label_path=args.nat_label_train,
                            transform=trans_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=True, num_workers=4, pin_memory=True)
    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)

    # init model, ResNet18() can be also used here for training
    model = resnet.to(device)
    #model = WideResNet(34, 10, 10).to(device)
    model = torch.nn.DataParallel(model)

    summary(model, input_size=(args.batch_size,3,32,32), verbose=2)
    
    cudnn.benchmark = True
    #next is original line
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        
        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print(64*'=')
        # eval_train(model, device, train_loader)
        _, test_accuracy = eval_test(model, device, test_loader)

        # save checkpoint
        if epoch % args.save_freq == 0:
            '''
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch)))
            '''

            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'best_model.pth'))
            print('save the model')

        print(64*'=')
    
    print("Saving resnet model ...")
    util.save_model(model, 'resnet18-100')
    print("Save Complete. Exiting ...")


def loadTest():
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    name = 'resnet18-100' #input("Name of model to load: ") #for now I'll hard code the only model I have trained
    model = torch.load(os.path.join(args.SAVE_MODEL_PATH, name))
    print(f"Model loaded: {name}")

    #https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min/notebook
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #https://github.com/kuangliu/pytorch-cifar/issues/19
    #stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)

    print(64*'=')
    _, test_accuracy = eval_test_w_adv(model, device, test_loader)
    print(64*'=')

if __name__ == '__main__':
    #main()
    loadTest()
    
    






