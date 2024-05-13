import os
import torch
from model_data_prepare import parse, prepare, init_Attacker
from evaluate import evalute_models

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

if __name__ == "__main__":
    args_attack = parse()

    # init attacker
    pgd_attack = init_Attacker(args_attack)

    # load the trained CMUA-Watermark
    if args_attack.global_settings.init_watermark_path:
        pgd_attack.up = torch.load(args_attack.global_settings.init_watermark_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Init the attacked models
    attack_dataloader, test_dataloader, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
    print("finished init the attacked models")

    print('The size of CMUA-Watermark: ', pgd_attack.up.shape)
    evalute_models(args_attack, test_dataloader, solver, attentiongan_solver, F, T, G, E, reference, pgd_attack)