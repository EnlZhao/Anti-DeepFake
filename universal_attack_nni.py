import os
from tqdm import tqdm

import torch
import torchvision.utils as vutils

from model_data_prepare import parse, prepare, init_Attacker
from evaluate import evalute_models
import nni

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def search():
    args_attack = parse()
    tuner_params = nni.get_next_parameter()
    args_attack.attacks.star_factor = float(tuner_params['star_factor'])
    args_attack.attacks.attention_factor = float(tuner_params['aggan_factor'])
    args_attack.attacks.HiSD_factor = float(tuner_params['HiSD_factor'])
    os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
    print("experiment dir is created")
    os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
    print("experiment config is saved")

    # Init the attacker
    pgd_attack = init_Attacker(args_attack)
    # Init the attacked models
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
        solver.test_universal_model_level_attack(img_a, c_org, pgd_attack)

        # attack attentiongan
        attentiongan_solver.test_universal_model_level_attack(img_a, c_org, pgd_attack)

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
            vutils.save_image(((x_trg + 1)/ 2).data, './adv_output_ori.jpg', padding=0)
        pgd_attack.universal_perturb_HiSD(img_a.to(device), transform, F, T, G, E, reference, x_trg+0.002, gen_models, mask)
        torch.save(pgd_attack.up, args_attack.global_settings.universal_perturbation_path)
        print('save the CMUA-Watermark')


    print('The size of CMUA-Watermark: ', pgd_attack.up.shape)
    HiDF_prop_dist, stargan_prop_dist, aggan_prop_dist = evalute_models(args_attack, test_dataloader, solver, attentiongan_solver, F, T, G, E, reference, pgd_attack)
    nni.report_intermediate_result(HiDF_prop_dist)
    nni.report_intermediate_result(stargan_prop_dist)
    nni.report_intermediate_result(aggan_prop_dist)
    nni.report_final_result(float((stargan_prop_dist+aggan_prop_dist+HiDF_prop_dist)/3))

if __name__=="__main__":
    search()
