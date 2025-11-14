import torch
import numpy as np


def safe_cross_entropy(p, logq, dim=-1):
    safe_logq = torch.where(p == 0, torch.ones_like(logq), logq)
    return -torch.sum(p * safe_logq, dim=dim)


def loss_func(logits_train, ideal_probs_train, R_train, baseline, lengths, gp_gamma_decay, entropy_gamma_decay,
              entropy_weight, device="cpu"):
    # ----- Getting shape -----
    (max_time_step, n_train, tokens_number,) = ideal_probs_train.shape
    mask_length_np = np.tile(np.arange(0, max_time_step), (n_train, 1)
                             ).astype(int) < np.tile(lengths,
                                                     (max_time_step, 1)).transpose()  # (n_train,max_time_step,)
    mask_length_np = mask_length_np.transpose().astype(float)  # (max_time_step, n_train,)
    mask_length = torch.tensor(mask_length_np, requires_grad=False).to(device)  # (max_time_step, n_train,)

    # ----- Entropy mask -----
    # Entropy mask (weighting differently along sequence dim)
    gp_gamma_decay = np.array(
        [gp_gamma_decay ** t for t in range(max_time_step)])  # (max_time_step,)
    gp_decay_mask_np = np.tile(gp_gamma_decay,
                               (n_train, 1)).transpose() * mask_length_np  # (max_time_step, n_train,)
    gp_decay_mask = torch.tensor(gp_decay_mask_np, requires_grad=False).to(
        device)  # (max_time_step, n_train,)

    # Entropy mask (weighting differently along sequence dim)
    entropy_gamma_decay = np.array(
        [entropy_gamma_decay ** t for t in range(max_time_step)])  # (max_time_step,)
    entropy_decay_mask_np = np.tile(entropy_gamma_decay,
                                    (n_train, 1)).transpose() * mask_length_np  # (max_time_step, n_train,)
    entropy_decay_mask = torch.tensor(entropy_decay_mask_np, requires_grad=False).to(
        device)  # (max_time_step, n_train,)

    # ----- Loss : Gradient Policy -----
    # Normalizing over action dim probs and logprobs
    logprobs = torch.nn.functional.log_softmax(logits_train, dim=2)  # (max_time_step, n_train, tokens_number,)
    # Sum over action dim # TODO:find why neglogp_per_step is nan
    neglogp_per_step = safe_cross_entropy(ideal_probs_train, logprobs, dim=2)  # (max_time_step, n_train,)
    neglogp_per_step = torch.where(torch.isinf(neglogp_per_step), torch.full_like(neglogp_per_step, 0.),
                                   neglogp_per_step)  # means model_prior+log_prior = all -inf, then action is randomly selected——lead to 1*-inf; these grad should not be transfered
    neglogp_per_step = torch.where(torch.isnan(neglogp_per_step), torch.full_like(neglogp_per_step, 0.),
                                   neglogp_per_step)
    # Sum over sequence dim
    neglogp = torch.sum(neglogp_per_step * mask_length, dim=0)  # (n_train,)  * gp_decay_mask
    # weights = 1. - torch.tanh(torch.linspace(-2, 2, steps=n_train)).to(device)
    loss_gp = torch.mean((R_train - baseline) * neglogp)  # Mean over training samples of batch:weights * 
    if loss_gp==0:
        print("loss_gp is 0, change baseline")
        baseline = baseline-0.1
        loss_gp = torch.mean((R_train - baseline) * neglogp)
    # loss_gp = torch.multiply(loss_gp, torch.tensor(R_train.shape[0] / 100.).to(device))

    # ----- Loss : maximum entropy reinforcement learning framework -----
    # Sum over action dim
    probs = torch.nn.functional.softmax(logits_train, dim=2)  # (max_time_step, n_train, tokens_number,)
    entropy_per_step = safe_cross_entropy(probs, logprobs, dim=2)  # (max_time_step, n_train,)
    # Sum over sequence dim
    # TODO: find why entropy_per_step is nan,先用下面的式子补救一下——补救失败
    entropy_per_step_no_nan = torch.where(torch.isnan(entropy_per_step), torch.full_like(entropy_per_step, 0.), entropy_per_step)
    entropy = torch.sum(entropy_per_step_no_nan * entropy_decay_mask, dim=0)  # (n_train,)
    # TODO: find why entropy is nan,先用下面的式子补救一下
    # entropy = torch.where(torch.isnan(entropy), torch.full_like(entropy, 0.), entropy) #发现2,0~57都是nan
    
    # loss_entropy = entropy_weight * torch.mean((R_train - baseline) * entropy) # need to minimize entropy
    loss_entropy = -entropy_weight * torch.mean(entropy)  # need to minimize entropy

    # ----- Loss-----
    loss = loss_gp + loss_entropy
    # print("loss_gp/loss_entropy ", loss_gp.cpu().detach().numpy())
    print("loss_gp/loss_entropy ", loss_gp.cpu().detach().numpy(), loss_entropy.cpu().detach().numpy())

    return loss
