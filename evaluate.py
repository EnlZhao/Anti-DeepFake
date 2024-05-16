import torch
import torch.nn.functional as Func

device = None
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
NUM_TEST = 20

def evalute_models(args_attack, test_dataloader, solver, attentiongan_solver, F, T, G, E, reference, pgd_attack):
    #  HiDF inference and evaluating
    l1_loss, l2_loss, l0_loss = 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (origin_img, origin_att, c_org) in enumerate(test_dataloader):
        if idx == NUM_TEST:
            break
        origin_img = origin_img.to(device)
        origin_att = origin_att.to(device)
        origin_att = origin_att.type(torch.float)
        
        with torch.no_grad():
            # clean
            c = E(origin_img)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen_noattack = G(c_trg)
            # adv
            c = E(origin_img + pgd_attack.up)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen = G(c_trg)
            mask = abs(gen_noattack - origin_img)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_loss += Func.l1_loss(gen, gen_noattack)
            l2_loss += Func.mse_loss(gen, gen_noattack)
            l0_loss += (gen - gen_noattack).norm(0)
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1

    print(f'HISD \t\t{n_samples} images. \tL1 loss: {l1_loss / n_samples}. \tL2 loss: {l2_loss / n_samples}. \tL0 loss: {l0_loss / n_samples}. \tdiff_proportion: {float(n_dist) * 100. / n_samples}%.')
    HiDF_prop_dist = float(n_dist) / n_samples

    # stargan inference and evaluating
    l1_loss, l2_loss, l0_loss = 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (origin_img, origin_att, c_org) in enumerate(test_dataloader):
        if idx == NUM_TEST:
            break
        origin_img = origin_img.to(device)
        origin_att = origin_att.to(device)
        origin_att = origin_att.type(torch.float)

        x_adv, x_noattack_list, x_fake_list = solver.test_universal_watermark(origin_img, c_org, pgd_attack.up, args_attack.stargan)
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            mask = abs(gen_noattack - origin_img)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_loss += Func.l1_loss(gen, gen_noattack)
            l2_loss += Func.mse_loss(gen, gen_noattack)
            l0_loss += (gen - gen_noattack).norm(0)
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
        

    print(f'StarGAN \t{n_samples} images. \tL1 loss: {l1_loss / n_samples}. \tL2 loss: {l2_loss / n_samples}. \tL0 loss: {l0_loss / n_samples}. \tdiff_proportion: {float(n_dist) * 100. / n_samples}%.')
    stargan_prop_dist = float(n_dist) / n_samples

    # AttentionGAN inference and evaluating
    l1_loss, l2_loss, l0_loss = 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (origin_img, origin_att, c_org) in enumerate(test_dataloader):
        if idx == NUM_TEST:
            break
        origin_img = origin_img.to(device)
        origin_att = origin_att.to(device)
        origin_att = origin_att.type(torch.float)
        x_adv, x_noattack_list, x_fake_list = attentiongan_solver.test_universal_watermark(origin_img, c_org, pgd_attack.up, args_attack.AttentionGAN)
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            mask = abs(gen_noattack - origin_img)
            mask = mask[0,0,:,:] + mask[0,1,:,:] + mask[0,2,:,:]
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            l1_loss += Func.l1_loss(gen, gen_noattack)
            l2_loss += Func.mse_loss(gen, gen_noattack)
            l0_loss += (gen - gen_noattack).norm(0)
            if (((gen*mask - gen_noattack*mask)**2).sum() / (mask.sum()*3)) > 0.05:
                n_dist += 1
            n_samples += 1
        

    print(f'AGGAN \t\t{n_samples} images. \tL1 loss: {l1_loss / n_samples}. \tL2 loss: {l2_loss / n_samples}. \tL0 loss: {l0_loss / n_samples}. \tdiff_proportion: {float(n_dist) * 100. / n_samples}%.')
    aggan_prop_dist = float(n_dist) / n_samples
    return HiDF_prop_dist, stargan_prop_dist, aggan_prop_dist