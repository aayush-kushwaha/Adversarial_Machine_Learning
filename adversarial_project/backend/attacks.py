import torch

from .model import device


def fgsm_attack(model, loss_fn, images, labels, epsilon):
    images = images.clone().detach().to(device)
    images.requires_grad = True

    output = model(images)
    loss = loss_fn(output, labels)
    model.zero_grad()
    loss.backward()

    adv = images + epsilon * images.grad.sign()
    return torch.clamp(adv, 0, 1)


def pgd_attack(model, loss_fn, images, labels, epsilon, alpha, iters):
    ori = images.clone().detach().to(device)
    adv = ori.clone().detach()

    for _ in range(iters):
        adv.requires_grad = True
        output = model(adv)
        loss = loss_fn(output, labels)
        model.zero_grad()
        loss.backward()

        adv = adv + alpha * adv.grad.sign()
        delta = torch.clamp(adv - ori, -epsilon, epsilon)
        adv = torch.clamp(ori + delta, 0, 1).detach()

    return adv
