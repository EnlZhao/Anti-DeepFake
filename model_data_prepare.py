import argparse
import json
from os.path import join
import torch
import torch.utils.data as data
from stargan.attacks import PGDAttack
from data import CelebA
from stargan.solver import Solver
from AttentionGAN.AttentionGAN_v1_multi.solver import Solver as AttentionGANSolver
from HiSD.inference import prepare_HiSD

device = None
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# init setting
def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

# init attacker
def init_Attacker(args_attack):
    pgd_attack = PGDAttack(model=None, device=device, epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return pgd_attack

# init AttGAN
def init_args(args_attack):
    with open(join('./stargan/setting.txt'), 'r') as f:
        args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    args.num_test = args_attack.global_settings.num_test
    args.gpu = args_attack.global_settings.gpu
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)
    return args

# init stargan
def init_stargan(args_attack, test_dataloader):
    return Solver(celeba_loader=test_dataloader, rafd_loader=None, config=args_attack.stargan)

# init attentiongan
def init_attentiongan(args_attack, test_dataloader):
    return AttentionGANSolver(celeba_loader=test_dataloader, rafd_loader=None, config=args_attack.AttentionGAN)

# init attack data
def init_attack_data(args_attack, attgan_args):
    test_dataset = CelebA(args_attack.global_settings.data_path, args_attack.global_settings.attr_path, args_attack.global_settings.img_size, 'test', attgan_args.attrs,args_attack.stargan.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=args_attack.global_settings.batch_size, num_workers=0,
        shuffle=False, drop_last=False
    )
    if args_attack.global_settings.num_test is None:
        print('Testing images:', len(test_dataset))
    else:
        print('Testing images:', min(len(test_dataset), args_attack.global_settings.num_test))
    return test_dataloader

# init inference data
def init_inference_data(args_attack, attgan_args):
    test_dataset = CelebA(args_attack.global_settings.data_path, args_attack.global_settings.attr_path, args_attack.global_settings.img_size, 'test', attgan_args.attrs,args_attack.stargan.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, num_workers=0,
        shuffle=False, drop_last=False
    )
    if args_attack.global_settings.num_test is None:
        print('Testing images:', len(test_dataset))
    else:
        print('Testing images:', min(len(test_dataset), args_attack.global_settings.num_test))
    return test_dataloader

def prepare():
    # prepare deepfake models
    args_attack = parse()
    args = init_args(args_attack)
    attack_dataloader = init_attack_data(args_attack, args)
    test_dataloader = init_inference_data(args_attack, args)
    solver = init_stargan(args_attack, test_dataloader)
    solver.restore_model(solver.test_iters)
    attentiongan_solver = init_attentiongan(args_attack, test_dataloader)
    attentiongan_solver.restore_model(attentiongan_solver.test_iters)
    transform, F, T, G, E, reference, gen_models = prepare_HiSD()
    print("Finished deepfake models initialization!")
    return attack_dataloader, test_dataloader, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models