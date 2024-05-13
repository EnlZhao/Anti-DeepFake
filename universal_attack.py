import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

from model_data_prepare import parse, prepare, init_Attacker
from evaluate import evalute_models

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

pgd_attack = init_Attacker(args_attack)

# 载入已有扰动
if args_attack.global_settings.init_watermark_path:
    pgd_attack.up = torch.load(args_attack.global_settings.init_watermark_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# init the attacker models
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
evalute_models(args_attack, test_dataloader, solver, attentiongan_solver, F, T, G, E, reference, pgd_attack)