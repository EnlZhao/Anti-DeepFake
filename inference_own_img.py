import sys
from PIL import Image

import torch
import torchvision.utils as vutils
import torch.nn.functional as Func
from torchvision import transforms

from model_data_prepare import parse, prepare, init_Attacker

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
    if sys.argv[1] == 'test':
        if args_attack.global_settings.universal_watermark_path:
            pgd_attack.up = torch.load(args_attack.global_settings.universal_watermark_path, map_location=device)
    elif args_attack.global_settings.init_watermark_path:
        pgd_attack.up = torch.load(args_attack.global_settings.init_watermark_path, map_location=device)

    # Init the attacked models
    _, test_dataloader, solver, attentiongan_solver, _, F, T, G, E, reference, _ = prepare()
    print("finished init the attacked models")

    # tf = transforms.Compose([
    #         # transforms.CenterCrop(170),
    #         transforms.Resize(args_attack.global_settings.img_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    # image = Image.open(sys.argv[2])
    # img = image.convert("RGB")
    # img = tf(img).unsqueeze(0)

    # stargan inference
    print(f"Run stargan inference")

    for idx, (img, att_a, c_org) in enumerate(test_dataloader):
        img = img.to(device)
        
        x_noattack_list, x_fake_list = solver.test_universal_watermark(img, c_org, pgd_attack.up, args_attack.stargan)
        
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
        
        # save original image
        out_file = './demo_results/stargan_original.jpg'
        vutils.save_image(img.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        for j in range(len(x_fake_list)):
            # save original image generated images
            gen_noattack = x_noattack_list[j]
            out_file = './demo_results/stargan_gen_{}.jpg'.format(j)
            vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, value_range=(-1., 1.))
            
            # save adversarial generated images
            gen = x_fake_list[j]
            out_file = './demo_results/stargan_advgen_{}.jpg'.format(j)
            vutils.save_image(gen, out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        break

    # AttentionGAN inference
    print(f"Run AttentionGAN inference")

    for idx, (img, _, c_org) in enumerate(test_dataloader):
        img = img.to(device)
        x_noattack_list, x_fake_list = attentiongan_solver.test_universal_watermark(img, c_org, pgd_attack.up, args_attack.AttentionGAN)
        
        # save original image
        out_file = './demo_results/attentiongan_original.jpg'
        vutils.save_image(img.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))

        for j in range(len(x_fake_list)):
            # save original image generated images
            gen_noattack = x_noattack_list[j]
            out_file = './demo_results/attentiongan_gen_{}.jpg'.format(j)
            vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, value_range=(-1., 1.))
            
            # save adversarial generated images
            gen = x_fake_list[j]
            out_file = './demo_results/attentiongan_advgen_{}.jpg'.format(j)
            vutils.save_image(gen, out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        break
    
    # HiSD inference
    print(f"Run HiSD inference")
    
    img = img.to(device)
    with torch.no_grad():
        # clean
        c = E(img)
        c_trg = c
        s_trg = F(reference, 1)
        c_trg = T(c_trg, s_trg, 1)
        gen_noattack = G(c_trg)

        # adv
        c = E(img + pgd_attack.up)
        c_trg = c
        s_trg = F(reference, 1)
        c_trg = T(c_trg, s_trg, 1)
        gen = G(c_trg)

        # save original image
        out_file = './demo_results/HiSD_original.jpg'
        vutils.save_image(img.cpu(), out_file, nrow=1, normalize=True, value_range=(-1., 1.))
        
        # save deepfake images generated from clean image
        out_file = './demo_results/HiSD_gen.jpg'
        vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, value_range=(-1., 1.))

        # save deepfake images generated from watermarked image
        gen = x_fake_list[j]
        out_file = './demo_results/HiSD_advgen.jpg'
        vutils.save_image(gen, out_file, nrow=1, normalize=True, value_range=(-1., 1.))

