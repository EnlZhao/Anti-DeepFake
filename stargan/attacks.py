import copy
import numpy as np
import torch
import torch.nn as nn

class PGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01, star_factor=0.3, attention_factor=0.3, HiSD_factor=1, feat=None, args=None):
        """
        PGD attacks
        epsilon: magnitude of attack
        step: iterations
        alpha: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.step = k
        self.alpha = a
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.loss_fn = nn.MSELoss().to(self.device)

        # Feature-level attack? Which layer?
        self.feat = feat

        # Universal perturbation
        self.up = None
        self.att_up = None
        self.attention_up = None
        self.star_up = None
        self.momentum = args.momentum
        
        #factors to control models' weights
        self.star_factor = star_factor
        self.attention_factor = attention_factor
        self.HiSD_factor = HiSD_factor

    def perturb(self, X_nat, y, c_trg):
        """
        Vanilla Attack.
        """
        X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)

        for i in range(self.step):
            X.requires_grad = True
            output, feats = self.model(X, c_trg)

            if self.feat:
                output = feats[self.feat]

            self.model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.alpha * grad.sign()
            
            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat

    def universal_perturb_stargan(self, X_nat, y, c_trg, model):
        """
        Vanilla Attack.
        """
        X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)

        for i in range(self.step):
            X.requires_grad = True
            output, feats = model(X, c_trg)
            if self.feat:
                output = feats[self.feat]
            model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad
            X_adv = X + self.alpha * grad.sign()

            if self.up is None:
                eta = torch.mean(
                    torch.clamp(self.star_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                    dim=0)
                self.up = eta
            else:
                eta = torch.mean(torch.clamp(self.star_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),dim=0)
                self.up = self.up * self.momentum + eta * (1 - self.momentum)
            X = torch.clamp(X_nat + self.up, min=-1, max=1).detach_()
            # print(loss.item())
        model.zero_grad()
        return X, X - X_nat

    def universal_perturb_attentiongan(self, X_nat, y, c_trg, model):
        """
        Vanilla Attack.
        Generate universal adversarial perturbation for AttentionGAN model.

        Args:
        - X_nat (torch.Tensor): The natural image tensor.
        - y (torch.Tensor): The true label tensor.
        - c_trg (torch.Tensor): The target class tensor.
        - model (torch.nn.Module): The AttentionGAN model.

        Returns:
        - X (torch.Tensor): The adversarial image tensor.
        - perturbation (torch.Tensor): The adversarial perturbation tensor.
        """
        X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        
        for i in range(self.step):
            X.requires_grad = True
            output, _, feats = model(X, c_trg)

            if self.feat:
                output = feats[self.feat]

            model.zero_grad()
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.alpha * grad.sign()
            if self.up is None:
                eta = torch.mean(
                    torch.clamp(self.attention_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                    dim=0)
                self.up = eta
            else:
                eta = torch.mean(
                    torch.clamp(self.attention_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                    dim=0)
                self.up = self.up * self.momentum + eta * (1 - self.momentum)
            X = torch.clamp(X_nat + self.up, min=-1, max=1).detach_()
        model.zero_grad()
        return X, X - X_nat

    def perturb_HiSD(self, X_nat, transform, F, T, G, E, reference, y, gen, mask):

        X = X_nat.clone().detach_() + self.up + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
           
        for i in range(self.step):
            X.requires_grad = True
            c = E(X)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)
            gen.zero_grad()

            loss = self.loss_fn(x_trg, y)
            loss.backward()
            
            grad = X.grad

            X_adv = X + self.alpha * grad.sign()
            if self.up is None:
                eta = torch.mean(
                    torch.clamp(self.HiSD_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                    dim=0)
                self.up = eta
            else:
                eta = torch.mean(
                    torch.clamp(self.HiSD_factor * (X_adv - X_nat), min=-self.epsilon, max=self.epsilon).detach_(),
                    dim=0)
                self.up = self.up * self.momentum + eta * (1 - self.momentum)
            X = torch.clamp(X_nat + self.up, min=-1, max=1).detach_()            
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