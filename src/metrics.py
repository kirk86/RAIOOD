import torch
import utils
import numpy as np
from scipy.special import digamma, gammaln, psi
from scipy.stats import dirichlet
from scipy.stats import entropy


def accuracy(mean, y_train):
    acc = torch.sum(torch.argmax(mean, 1) == y_train.squeeze()
                    ).numpy() / float(len(mean))

    print("Accuracy {0:.2f}%".format(acc * 100))
    # return acc


def get_entropy(mean):

    entropys = []
    for row in mean:
        ent = entropy(row.detach())
        entropys.append(ent)
    return np.array(entropys)


def calc_dirichlet_differential_entropy(alphas, epsilon=1e-8):
    # Calculate Expected Entropy of categorical distribution under dirichlet Prior.
    # Higher means more uncertain
    alphas = np.asarray(alphas, dtype=np.float64) + epsilon
    diff_entropy = np.asarray([dirichlet(alpha).entropy() for alpha in alphas])
    return diff_entropy


def dirichlet_uncertainty(logits, eps=1.e-10):
    """
    :param logits:
    :param epsilon:
    :return:
    test
    """

    logits = np.asarray(logits)
    probs = utils.softmax(logits)
    conf = probs.max(axis=1)
    alphas = np.exp(logits - np.max(logits, axis=-1, keepdims=True)) + eps
    alpha0 = np.sum(alphas, axis=-1, keepdims=True)
    # probs = alphas / alpha0

    entropy_of_conf = -np.sum(probs * np.log(probs + eps), axis=1)
    expected_entropy = -np.sum((alphas/alpha0) * (digamma(alphas + 1) - digamma(alpha0 + 1.0)),
                               axis=1)
    mutual_info = entropy_of_conf - expected_entropy

    epkl = np.squeeze((alphas.shape[1] - 1.0) / (alpha0 + eps))

    dentropy = np.sum(gammaln(alphas) - (alphas - 1.0) * (digamma(alphas) - digamma(alpha0)),
                      axis=1, keepdims=True) \
        - gammaln(alpha0)

    uncertainty = {
        'confidence': conf,
        'entropy_of_conf': entropy_of_conf,
        'expected_entropy': expected_entropy,
        'mutual_information': mutual_info,
        'EPKL': epkl,
        'differential_entropy': np.squeeze(dentropy),
        }

    return uncertainty


def differential_entropy(logits):
    alpha_c = np.exp(logits)
    alpha_c = np.clip(alpha_c, 10e-10, 10e10)    
    alpha_0 = np.sum(alpha_c, axis=1)
    gammaln_alpha_c = gammaln(alpha_c)
    gammaln_alpha_0 = gammaln(alpha_0)
    
    psi_alpha_c = psi(alpha_c)
    psi_alpha_0 = psi(alpha_0)
    psi_alpha_0 = np.expand_dims(psi_alpha_0, axis=1)
    
    temp_mat = np.sum((alpha_c - 1) * (psi_alpha_c - psi_alpha_0), axis=1)
    
    metric = np.sum(gammaln_alpha_c, axis=1) - gammaln_alpha_0 - temp_mat
    return metric


def mutual_info(logits):
    logits = logits.astype('float64')
    alpha_c = np.exp(logits)
    alpha_c = np.clip(alpha_c, 10e-10, 10e10)    
    alpha_0 = np.sum(alpha_c, axis=1, keepdims=True)
 
    psi_alpha_c = psi(alpha_c + 1)
    psi_alpha_0 = psi(alpha_0 + 1)
    alpha_div = alpha_c / alpha_0
    
    temp_mat = np.sum(-alpha_div * (np.log(alpha_c) - psi_alpha_c), axis=1)
    metric = temp_mat + np.squeeze(np.log(alpha_0) - psi_alpha_0)
    return metric


def _get_prob(logits):
    alpha_c = np.exp(logits)
    alpha_c = np.clip(alpha_c, 10e-40, 10e40)
    alpha_0 = np.sum(alpha_c, axis=1)
    alpha_0 = np.expand_dims(alpha_0, axis=1)
    
    return (alpha_c/ alpha_0)


def entropy(logits):
    prob = _get_prob(logits)
    exp_prob = np.log(prob)
    
    ent = -np.sum(prob * exp_prob, axis=1)
    return ent


def max_prob(logits):
    prob = _get_prob(logits)
    metric = np.max(prob, axis=1)
    return metric


def uncertainty_metrics(logits):
    max_prob_ = max_prob(logits)
    entropy_ = entropy(logits)
    mutual_info_ = mutual_info(logits)
    diff_entropy = differential_entropy(logits)
    metrics = {    
        'confidence': max_prob_,
        'entropy_of_conf': entropy_,
        'mutual_information': mutual_info_,
        'diff_entropy': diff_entropy,
    }
    return metrics