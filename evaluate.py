
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_TEST = 20

def evaluate_multiple_models(args_attack, test_dataloader, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, pgd_attack):
    #  HiDF inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        if idx == NUM_TEST:
            break
        img_a = img_a.to(device)
        
        with torch.no_grad():
            # clean
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen_noattack = G(c_trg)
            # adv
            c = E(img_a + pgd_attack.up)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen = G(c_trg)
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_error += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_error += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1

    print('HiDF {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))
    HiDF_prop_dist = float(n_dist) / n_samples

    # stargan inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        if idx == NUM_TEST:
            break
        img_a = img_a.to(device)
        att_a = att_a.to(device)
        att_a = att_a.type(torch.float)
        x_noattack_list, x_fake_list = solver.test_universal_model_level(idx, img_a, c_org, pgd_attack.up, args_attack.stargan)
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_error += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_error += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
        

    print('stargan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))
    stargan_prop_dist = float(n_dist) / n_samples

    # AttentionGAN inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        if idx == NUM_TEST:
            break
        img_a = img_a.to(device)
        att_a = att_a.to(device)
        att_a = att_a.type(torch.float)
        x_noattack_list, x_fake_list = attentiongan_solver.test_universal_model_level(idx, img_a, c_org, pgd_attack.up, args_attack.AttentionGAN)
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            mask = abs(gen_noattack - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_error += torch.nn.functional.l1_loss(gen, gen_noattack)
            l2_error += torch.nn.functional.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
        

    print('attentiongan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))
    aggan_prop_dist = float(n_dist) / n_samples
    return HiDF_prop_dist, stargan_prop_dist, aggan_prop_dist