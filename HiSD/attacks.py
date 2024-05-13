import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torchvision.utils as vutils
import torch
import torch.nn as nn
from torch.autograd import Variable

device = None
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

class LinfPGDAttack(object):
    def __init__(self, model=None, device=device, epsilon=0.05, k=100, a=0.01, star_factor=1, attention_factor=0.3, att_factor=2, feat=None, args=None):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # Feature-level attack? Which layer?
        self.feat = feat

        # PGD or I-FGSM?
        self.rand = True

        # Universal perturbation
        self.up = None
        self.att_up = None
        self.attention_up = None
        self.star_up = None
        self.momentum = 0.5
        
        #factors to control models' weights
        self.star_factor = star_factor
        self.attention_factor = attention_factor
        self.att_factor = att_factor


    def universal_perturb_HiSD(self, X_nat, transform, F, T, G, E, device, reference, y, gen):

        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
           
        for i in range(self.k):
            X.requires_grad = True
            c = E(X)
            c_trg = c
            s_trg = F(reference, 0)
            c_trg = T(c_trg, s_trg, 0)
            x_trg = G(c_trg)
            # model.zero_grad()
            gen.zero_grad()

            loss = self.loss_fn(x_trg, y)
            loss.backward()
            # print(i, loss.item())
            grad = X.grad

            X_adv = X + self.a * grad.sign()
            if self.up is None:
                eta = torch.mean(
                    torch.clamp(self.star_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                    dim=0)
                self.up = eta
            else:
                eta = torch.mean(
                    torch.clamp(self.star_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                    dim=0)
                self.up = self.up * self.momentum + eta * (1 - self.momentum)
            X = torch.clamp(X_nat + self.up, min=-1, max=1).detach_()
            if i % 100 == 0:
                vutils.save_image(((x_trg + 1)/ 2).data, './adv_output.jpg', padding=0)
            
        gen.zero_grad()
        return X, X - X_nat
        

def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res


def perturb_batch(X, y, c_trg, model, adversary):
    # Perturb batch function for adversarial training
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()
    adversary.model = model_cp
    X_adv, _ = adversary.perturb(X, y, c_trg)
    return X_adv
