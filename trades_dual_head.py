import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def rev_smoothing(w_output, y_true, alpha=1.0):

    w_onehot = F.one_hot(y_true, num_classes=10)
    #maxx = torch.max(w_output, dim=1).values
    maxx = w_output.gather(1, y_true.view(-1,1))
    
    betaa = (alpha * maxx) / (1.0001 - maxx)
    labda = 1 + betaa
    
    return w_output * labda - betaa * w_onehot * maxx


def trades_loss(model,
                x_natural,
                y_true, #y_target,
                optimizer,
                step_size=0.007,
                epsilon=0.031,
                perturb_steps=10, beta0=1.0,
                beta1=1.0, beta2=1.0, beta3=0.1,
                distance='l_inf', epoch=0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)

    # adjust learning rate / loss para. according to epoch
    for pt_epoch in (60,70,80,90,100):
        if epoch >= pt_epoch:
            beta0 = 0.95 * beta0
            beta1 = 0.95 * beta1
            beta2 = 0.8 * beta2
            beta3 = 0.8 * beta3

    # generate adversarial example
    x_adv = x_natural.detach() + epsilon*torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv)[0], dim=1),
                                       F.softmax(model(x_natural)[0], dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits_natural, aux_logits_natural = model(x_natural)
    logits_adv, aux_logits_adv = model(x_adv)
    #adv_lbl = torch.argmax(logits_adv)

    y_nat = logits_natural.max(dim=1)[1]
    y_adv = logits_adv.max(dim=1)[1]

    flag_success = y_true.eq(y_nat) * ~y_true.eq(y_adv)
    flag_failure = y_true.eq(y_nat) * y_true.eq(y_adv)

    ##
    lgs_s_nat = logits_natural[flag_success]
    lgs_f_nat = logits_natural[flag_failure]
    
    lgs_s_adv = logits_adv[flag_success]
    lgs_f_adv = logits_adv[flag_failure]
    
    lgs_s_nat_aux = aux_logits_natural[flag_success]
    lgs_f_nat_aux = aux_logits_natural[flag_failure]

    lgs_s_adv_aux = aux_logits_adv[flag_success]
    lgs_f_adv_aux = aux_logits_adv[flag_failure]

    y_s = y_true[flag_success]
    y_f = y_true[flag_failure]
    ##

    len_succ = lgs_s_nat.shape[0]
    len_fail = lgs_f_nat.shape[0]
    
    loss_natural = F.cross_entropy(logits_natural, y_true)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                            F.softmax(logits_natural, dim=1))

    loss_aux1 = loss_aux2 = 0.0
    if len_succ:
        loss_aux1 = (1.0 / batch_size) * criterion_kl(F.log_softmax(lgs_s_adv_aux, dim=1),
                                rev_smoothing(F.softmax(lgs_s_adv, dim=1), y_s, alpha=1.1))
    if len_fail:
        loss_aux2 = (1.0 / batch_size) * criterion_kl(F.log_softmax(lgs_f_adv_aux, dim=1),
                                rev_smoothing(F.softmax(lgs_f_adv, dim=1), y_f, alpha=1.0))

    loss_aux_nat = (1.0 / batch_size) * criterion_kl(F.log_softmax(aux_logits_natural, dim=1),
                            rev_smoothing(F.softmax(logits_natural, dim=1), y_true, alpha=0.9))

    loss = beta0 * loss_natural + beta1 * loss_robust + beta2 * (loss_aux1 + loss_aux2) + beta3 * loss_aux_nat

    return loss
