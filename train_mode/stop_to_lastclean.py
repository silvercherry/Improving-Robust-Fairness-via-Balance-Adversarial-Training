from models import *
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# based on the https://github.com/zjfheart/Friendly-Adversarial-Training
def stop_to_lastclean(model, data, target, print_flag, step_size, epsilon, perturb_steps,randominit_type,loss_fn,tau=10,rand_init=True,omega=0):
    model.eval()

    K = perturb_steps
    count = 0
    output_target = []
    output_adv = []
    output_natural = []

    control = (torch.ones(len(target))).cuda()

    if rand_init:
        if randominit_type == "normal_distribution_randominit":
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        if randominit_type == "uniform_randominit":
            iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
    else:
        iter_adv = data.cuda().detach()

    iter_clean_data = data.cuda().detach()
    iter_index_flag = torch.tensor([num for num in range(len(iter_clean_data))])
    iter_target = target.cuda().detach()
    output_iter_clean_data = model(data)

    tmp_adv = iter_adv
    if print_flag:
        print('--------begin while-----------')
    while K>0:
        iter_adv.requires_grad_()
        output = model(iter_adv)
        pred = output.max(1, keepdim=True)[1]
        output_index = [] 
        iter_index = [] 

        for idx in range(len(pred)):
            if pred[idx] != iter_target[idx]:
                output_index.append(idx)
            else:
                iter_index.append(idx)

        if len(output_index) != 0:
            if len(output_target) == 0:
                output_adv = tmp_adv[output_index].reshape(-1, 3, 32, 32).cuda()
                output_natural = iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()
                output_target = iter_target[output_index].reshape(-1).cuda()
                output_index_flag = iter_index_flag[output_index].reshape(-1)
            else:
                output_adv = torch.cat((output_adv, tmp_adv[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_index_flag = torch.cat((output_index_flag, iter_index_flag[output_index].reshape(-1)), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).cuda()), dim=0)
        if print_flag:
            print(10-K, output_target)

        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction='mean')(output, iter_target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        if len(iter_index) != 0:
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            tmp_adv = iter_adv
            iter_clean_data = iter_clean_data[iter_index]
            iter_index_flag = iter_index_flag[iter_index]
            iter_target = iter_target[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()

            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().cuda()
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, 0, 1)

            count += len(iter_target)
        else:
            output_adv = output_adv.detach()
            return output_adv, output_target, output_natural, output_index_flag
        K = K-1
    if print_flag:
        print('--------end while-----------')
        print(len(output_target))


    if len(output_target) == 0:
        output_target = iter_target.reshape(-1).squeeze().cuda()
        output_adv = iter_adv.reshape(-1, 3, 32, 32).cuda()
        output_natural = iter_clean_data.reshape(-1, 3, 32, 32).cuda()
        output_index_flag = iter_index_flag.reshape(-1).squeeze()
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, 32, 32)), dim=0).cuda()
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().cuda()
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, 32, 32).cuda()),dim=0).cuda()
        output_index_flag = torch.cat((output_index_flag, iter_index_flag.reshape(-1)), dim=0)
    output_adv = output_adv.detach()
    if print_flag:
        print(output_target)
    return output_adv, output_target, output_natural, output_index_flag
