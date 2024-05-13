import argparse
import json
import os
from os.path import join

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

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

args_attack = parse()
# print(args_attack)
# os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
# print("experiment dir is created")
# os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
# print("experiment config is saved")

# init attacker
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epsilon=args_attack.attacks.epsilon, step=args_attack.attacks.step, alpha=args_attack.attacks.alpha, star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return pgd_attack

pgd_attack = init_Attack(args_attack)

# load the trained CMUA-Watermark
if args_attack.global_settings.init_watermark_path:
    pgd_attack.up = torch.load(args_attack.global_settings.init_watermark_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Init the attacked models
attack_dataloader, test_dataloader, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
print("finished init the attacked models")

print('The size of CMUA-Watermark: ', pgd_attack.up.shape)
evaluate_multiple_models(args_attack, test_dataloader, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, pgd_attack)