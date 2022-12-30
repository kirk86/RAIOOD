import torch

F = torch.nn.functional


def cross_entropy(logits, y, batch_size):
    no_cosine = torch.tensor([0.0]).to(logits.device)
    if len(logits) > batch_size:
        logits, logits_ood = torch.split(logits, batch_size)
    loss = F.cross_entropy(logits, y)
    probs = F.softmax(logits, dim=1)
    y_hat = probs.argmax(dim=1, keepdim=True)
    return loss, y_hat, no_cosine


def contrastive_regularized(logits, y, batch_size, lamda=-1.0):
    probs, probs_ood = F.softmax(logits, dim=1).split(batch_size)
    CE, _, _ = cross_entropy(logits, y, batch_size)
    cosine = F.cosine_similarity(probs, probs_ood, dim=1).mean()
    loss = 2 * CE + (lamda * cosine)
    y_hat =  probs.argmax(dim=1)
    return loss, y_hat, cosine


def contrastive_ranking(logits, y, batch_size, gamma=torch.tensor([1.0])):
    n_classes = len(y.unique())
    zero = torch.tensor([0.0]).to(logits.device)
    gamma = gamma.to(logits.device)
    probs, probs_ood = F.softmax(logits, dim=1).split(batch_size)
    cosine = F.cosine_similarity(probs, probs_ood, dim=1).mean()
    l1 = 0.005 * (probs_ood - 1.0/n_classes).abs().sum()
    l2 = 0.007 * ((probs[range(batch_size), y] - 0.9954)**2).sum()
    margin = torch.maximum(zero, gamma + cosine)
    loss = margin + l1 + l2
    y_hat = probs.argmax(dim=1)
    return loss-gamma, y_hat, cosine