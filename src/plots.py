import torch
import numpy as np
import matplotlib.pyplot as plt
from src import metrics

nn = torch.nn
F = torch.nn.functional

def plot_x(x, ax=None, labels=None):
    if ax is None:
        return plt.scatter(x[:, 0], x[:, 1], c=labels)
    return ax.scatter(x[:, 0], x[:, 1], c=labels)


def plot_entropys(ent, points):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c=ent)
    plt.suptitle('Entropy distribution (lower more certain)')


def plot_data(*args, ax=None):
    plt.figure()
    for arg in args:
        plot_x(arg, ax)

def plot_train_stats(epochs, losses, accuracies):
    plt.plot(range(epochs), losses, label='loss')
    plt.plot(range(epochs), accuracies, label='acc')
    plt.title("Loss vs Accuracy on Train")
    plt.legend()
    plt.grid()


def plot_prediction_uncertainty(net, points):
    points_torch = torch.from_numpy(points).float()
    # Given toy dataset
    mean, alpha, precision = net(points_torch)

    N = np.int_(np.sqrt(points.shape[0]))
    max_prod = np.max(mean.detach().numpy(), 1).reshape((N, N))
    x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), N)
    y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), N)

    plt.figure()
    plt.contourf(x, y, max_prod, levels=20, cmap='Blues')
    plt.suptitle('Max class probability (higher more certain)')


def create_meshgrid(data):
    x_range = (data[:, 0].min() - 1, data[:, 0].max() + 1)
    y_range = (data[:, 1].min() - 1, data[:, 1].max() + 1)

    x = np.arange(x_range[0], x_range[1], 0.1)
    y = np.arange(y_range[0], y_range[1], 0.1)
    x_coord, y_coord = np.meshgrid(x, y)
    grid = np.c_[x_coord.ravel(), y_coord.ravel()]
    return grid, x_coord, y_coord


def plot_contour(X, Y, x, y, z, title, ax=None, fig=None):
    # N = np.int_(np.sqrt(points.shape[0]))
    # z = z.reshape((N, N))
    # x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), N)
    # y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), N)

    if ax is None or fig is None:
        plt.figure()
        cs = plt.contourf(x, y, z)
        ax.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
        cbar = plt.colorbar(cs)
        plt.suptitle(title)
    else:
        cs = ax.contourf(x, y, z)
        ax.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
        fig.colorbar(cs, ax=ax)
        ax.title.set_text(title)
        # return cs


def plot_net(logits, X, y, xx, yy, axs, fig, sigma, a0=None, method=None):

    if a0 is not None:
        conf_title = 'Confidence $\\alpha={}$'.format(a0)
        ent_title = 'Entropy $\\alpha={}$'.format(a0)
        mutual_info_title = 'Mutual Information $\\alpha={}$'.format(a0)
        diff_ent_title = 'Differential Entropy $\\alpha={}$'.format(a0)
        epkl_title = 'EPKL $\\alpha={}$'.format(a0)

    else:
        conf_title = 'Confidence'
        ent_title = 'Entropy'
        mutual_info_title = 'Mutual Information'
        diff_ent_title = 'Diff. Entropy'
        epkl_title = 'EPKL'

    if axs[0] is not None:
#         plot_train(X, y.squeeze(), axs[0])
        axs[0].scatter(X[:, 0], X[:, 1], c=y.squeeze())

    uncert_metrics = metrics.dirichlet_uncertainty(logits)
    confidence = uncert_metrics['confidence'].reshape(xx.shape)
    entropy = uncert_metrics['entropy_of_conf'].reshape(xx.shape)
    mutual_info = uncert_metrics['mutual_information'].reshape(xx.shape)
    diff_entropy = 1 - np.log(np.abs(uncert_metrics['differential_entropy']) + 1.e-6).reshape(xx.shape)

    plot_contour(X, y, xx, yy, confidence, conf_title, ax=axs[1], fig=fig)
    plot_contour(X, y, xx, yy, entropy, ent_title, ax=axs[2], fig=fig)
    plot_contour(X, y, xx, yy, mutual_info, mutual_info_title, ax=axs[3], fig=fig)
#     plot_contour(X, y, xx, yy, diff_entropy, diff_ent_title, ax=axs[4], fig=fig)
    # diff_val = np.log(np.abs(uncert_metrics['differential_entropy'])) - uncert_metrics['mutual_information']


def plot_loss(Xi, Yi, Zi):
    from matplotlib.colors import LightSource
    fig = plt.figure(figsize=(5,5))
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ls = LightSource(azdeg=0, altdeg=200)
    rgb = ls.shade(Zi, plt.cm.coolwarm)

    surf = ax.plot_surface(Xi, Yi, Zi, rstride=1, cstride=1, linewidth=0,
                       antialiased=True, facecolors=rgb)
    ax.set_title("Projection of loss along 2 dim. in input space\n(actual grad direction, random direction).")
    ax.set_xlabel('random direction')
    ax.set_xticks([])
    ax.set_ylabel('grad direction')
    ax.set_yticks([])
    ax.set_zlabel('loss')
