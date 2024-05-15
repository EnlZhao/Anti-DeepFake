import os
from tqdm import tqdm
import torch
from model_data_prepare import parse, prepare, init_Attacker
from evaluate import evalute_models

device = None
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.backends.cudnn.enabled = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    args_attack = parse()

    # create the experiment dir
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(f"{args_attack.global_settings.results_path}/results{args_attack.attacks.momentum}"):
        os.makedirs(f"{args_attack.global_settings.results_path}/results{args_attack.attacks.momentum}")

    os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
    print("experiment dir is created")

    os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
    print("experiment config is saved")

    pgd_attack = init_Attacker(args_attack)

    # 载入已有扰动
    if args_attack.global_settings.init_watermark_path:
        pgd_attack.up = torch.load(args_attack.global_settings.init_watermark_path, map_location=device)

    # init the attacker models
    attack_dataloader, test_dataloader, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
    print(f"finished init the attacked models, only attack {args_attack.global_settings.num_test // args_attack.global_settings.batch_size} epochs")

    # attacking models
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size >= args_attack.global_settings.num_test:
            break
        img_a = img_a.to(device)
        att_a = att_a.to(device)
        att_a = att_a.type(torch.float)

        # attack stargan
        solver.universal_perturb(img_a, c_org, pgd_attack)

        # attack attentiongan
        attentiongan_solver.universal_perturb(img_a, c_org, pgd_attack)

        # attack HiSD
        with torch.no_grad():
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)
            mask = abs(x_trg - img_a)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        pgd_attack.perturb_HiSD(img_a, transform, F, T, G, E, reference, x_trg+0.002, gen_models, mask)
        path, file_name = os.path.split(args_attack.global_settings.universal_watermark_path)
        pt_file = os.path.join(path, '{}_'.format(idx) + file_name)
        torch.save(pgd_attack.up, pt_file)
        print('save the Watermark')

    # evalute_models(args_attack, test_dataloader, solver, attentiongan_solver, F, T, G, E, reference, pgd_attack)