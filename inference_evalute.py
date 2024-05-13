import os
import torch
from model_data_prepare import parse, prepare, init_Attacker
from evaluate import evalute_models

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = None
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

if __name__ == "__main__":
    args_attack = parse()

    # init attacker
    pgd_attack = init_Attacker(args_attack)

    # load the trained CMUA-Watermark
    if args_attack.global_settings.init_watermark_path:
        pgd_attack.up = torch.load(args_attack.global_settings.init_watermark_path, map_location=device)

    # Init the attacked models
    _, test_dataloader, solver, attentiongan_solver, _, F, T, G, E, reference, _ = prepare()
    print("finished init the attacked models")

    print('The size of CMUA-Watermark: ', pgd_attack.up.shape)
    evalute_models(args_attack, test_dataloader, solver, attentiongan_solver, F, T, G, E, reference, pgd_attack)