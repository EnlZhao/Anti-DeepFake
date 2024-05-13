import argparse
import json
import os
from os.path import join
from tqdm import tqdm

import torch
import torch.nn.functional as F

import attacks

from model_data_prepare import prepare
from evaluate import evaluate_multiple_models

class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

        
    return args_attack

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.backends.cudnn.enabled = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args_attack = parse()
print(args_attack)
os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
print("experiment dir is created")
os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
print("experiment config is saved")

# init the attacker
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                       epsilon=args_attack.attacks.epsilon, step=args_attack.attacks.step, alpha=args_attack.attacks.alpha, 
                                       star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, 
                                       att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return pgd_attack


pgd_attack = init_Attack(args_attack)

# 载入已有扰动
if args_attack.global_settings.init_watermark_path:
    pgd_attack.up = torch.load(args_attack.global_settings.init_watermark_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# init the attacker models
# attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
attack_dataloader, test_dataloader, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
print("finished init the attacked models, only attack 2 epochs")

# attacking models
for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
    if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
        break
    img_a = img_a.to(device)
    att_a = att_a.to(device)
    att_a = att_a.type(torch.float)

    # attack stargan
    solver.test_universal_model_level_attack(idx, img_a, c_org, pgd_attack)

    # attack attentiongan
    attentiongan_solver.test_universal_model_level_attack(idx, img_a, c_org, pgd_attack)

    # attack HiSD
    with torch.no_grad():
        c = E(img_a)
        c_trg = c
        s_trg = F(reference, 1)
        c_trg = T(c_trg, s_trg, 1)
        x_trg = G(c_trg)
        mask = abs(x_trg - img_a)
        mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
        mask[mask>0.5] = 1
        mask[mask<0.5] = 0
    pgd_attack.universal_perturb_HiSD(img_a, transform, F, T, G, E, reference, x_trg+0.002, gen_models, mask)

    torch.save(pgd_attack.up, args_attack.global_settings.universal_perturbation_path)
    print('save the CMUA-Watermark')

print('The size of CMUA-Watermark: ', pgd_attack.up.shape)
attgan = None
evaluate_multiple_models(args_attack, test_dataloader, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, pgd_attack)