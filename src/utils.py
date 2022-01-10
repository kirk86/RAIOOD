import torch
import numpy as np
from tqdm.autonotebook import tqdm
#from augmentations import polar_transform, create_patches

nn = torch.nn
F = torch.nn.functional

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        #n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i:i+n].view(tensor.shape))
        i += n
    return outList


def to_torch(points):
    points_torch = torch.from_numpy(points).float()
    return points_torch


def generate_ood_multiclass(N, sigma, use_torch=False):
#     mu1 = np.array([0, 4/3])*3
#     mu2 = np.array([2, 4/3])*3
#     mu3 = np.array([1, -np.sqrt(3)])*3
    mu4 = np.array([-2/3, 1])*4
    mu5 = np.array([1, 1])*2
    mu6 = np.array([2, 1])*4
    mu7 = np.array([3.5, 8])

    def cov(_alpha): return np.array([[1, 0], [0, 1]]) * _alpha

    alpha = sigma
#     alpha1 = cov(alpha)
#     alpha2 = cov(alpha)
#     alpha3 = cov(alpha)
    alpha4 = cov(alpha)
    alpha5 = cov(alpha)
    alpha6 = cov(alpha) # np.array([[3, -2], [-2, 3]])
    alpha7 = cov(alpha)

    def gen_normal(mu, alpha):
        return np.random.multivariate_normal(
            mu, alpha, size=N)

#     x1 = gen_normal(mu1, alpha1)
#     x2 = gen_normal(mu2, alpha2)
#     x3 = gen_normal(mu3, alpha3)
    x4 = gen_normal(mu4, alpha4)
    x5 = gen_normal(mu5, alpha5)
    x6 = gen_normal(mu6, alpha6)
    x7 = gen_normal(mu7, alpha7)

#     y1 = np.zeros((N, 1))
#     y2 = np.ones((N, 1))
#     y3 = np.ones((N, 1))*2
    y4 = np.ones((N, 1))*3
    y5 = np.ones((N, 1))*4
    y6 = np.ones((N, 1))*5
    y7 = np.ones((N, 1))*6

#     x1 = np.concatenate((x1, y1), axis=1)
#     x2 = np.concatenate((x2, y2), axis=1)
#     x3 = np.concatenate((x3, y3), axis=1)
#     x4 = np.concatenate((x4, y4), axis=1)
#     x5 = np.concatenate((x5, y5), axis=1)
#     x6 = np.concatenate((x6, y6), axis=1)
#     x7 = np.concatenate((x7, y7), axis=1)

#     X = np.vstack((x1, x2, x3, x4, x5, x6, x7))
#     np.random.shuffle(X)
    X_ood = np.vstack((x4, x5, x6, x7))
    np.random.shuffle(X_ood)

    if use_torch:
        return torch.from_numpy(X_ood).float()
    return X_ood


def generate_toy_dataset(N, sigma, use_torch=False):
    mu1 = np.array([0, 0])*3
    mu2 = np.array([2, 0])*3
    mu3 = np.array([1, np.sqrt(3)])*3

    def cov(_alpha): return np.array([[1, 0], [0, 1]]) * _alpha

    alpha = sigma
    alpha1 = cov(alpha)
    alpha2 = cov(alpha)
    alpha3 = cov(alpha)

    def gen_normal(mu, alpha):
        return np.random.multivariate_normal(
            mu, alpha, size=N)

    x1 = gen_normal(mu1, alpha1)
    x2 = gen_normal(mu2, alpha2)
    x3 = gen_normal(mu3, alpha3)

    y1 = np.zeros((N, 1))
    y2 = np.ones((N, 1))
    y3 = np.ones((N, 1))*2

    x1 = np.concatenate((x1, y1), axis=1)
    x2 = np.concatenate((x2, y2), axis=1)
    x3 = np.concatenate((x3, y3), axis=1)

    X = np.vstack((x1, x2, x3))
    np.random.shuffle(X)

    X_ood = generate_ood(X.shape[0], X[:, :2], 5)

    if use_torch:
        return torch.from_numpy(X).float()
    return X, X_ood


def generate_points(x_train, y_train, N=100, r=10):
    x_range = (x_train[:, 0].min()-r, x_train[:, 0].max()+r)
    y_range = (x_train[:, 1].min()-r, x_train[:, 1].max()+r)

    x = np.linspace(x_range[0], x_range[1], N)
    y = np.linspace(y_range[0], y_range[1], N)
    coord = np.array([[(i, j) for i in x] for j in y]).reshape(-1, 2)
    return coord


def generate_ood(N, x_train, dist, dim=2):
    r = dist
    x_range = (x_train[:, 0].min()-r, x_train[:, 0].max()+r)
    v = np.random.uniform(x_range[0], x_range[1], size=(N, dim))
    return v


def gmm_sample(n_samples, components, mu, sigma=0.5, d=2):
    # mu = [np.array([0, -2]), np.array([0, 0])...] list of mus equal length to components
    assert n_samples % components == 0
    sample_values = []
    for i in range(components):
        sample_values.append(np.random.randn(n_samples // components, d)*sigma + mu[i])
    samples  = np.concatenate(sample_values, axis=0)
    return samples


def create_meshgrid(data):
    x_range = (data[:, 0].min() - 5, data[:, 0].max() + 10)
    y_range = (data[:, 1].min() - 2, data[:, 1].max() + 10)
    # x_range = (-5, 10)
    # y_range = (-2, 10)

    x = np.arange(x_range[0], x_range[1], 0.1)
    y = np.arange(y_range[0], y_range[1], 0.1)
    x_coord, y_coord = np.meshgrid(x, y)
    return np.c_[x_coord.ravel(), y_coord.ravel()], x_coord, y_coord


def draw_loss(model, X, y, epsilon=0.1, device="cuda:0"):
    """
    X should be just an entry from the corresponding batch, e.g. X.shape = (1, 3, 32, 32)
    y should be just an entry from the corresponding batch, e.g. y.shape = (1,)
    """
    Xi, Yi = np.meshgrid(np.linspace(-epsilon, epsilon, 50), np.linspace(-epsilon, epsilon, 50))

    def grad_at_delta(delta):
        delta.requires_grad_(True)
        nn.CrossEntropyLoss()(model(X + delta), y).backward()
        return delta.grad.detach().sign().view(-1).cpu().numpy()

    dir1 = grad_at_delta(torch.zeros_like(X, requires_grad=True))
    delta2 = torch.zeros_like(X, requires_grad=True)
    delta2.data = torch.tensor(dir1).view_as(X).to(device)
    dir2 = grad_at_delta(delta2)
    np.random.seed(0)
    dir2 = np.sign(np.random.randn(dir1.shape[0]))

    all_deltas = torch.tensor((np.array([Xi.flatten(), Yi.flatten()]).T @ 
                              np.array([dir2, dir1])).astype(np.float32)).to(device)
    data = all_deltas.view(-1, 3, 32, 32) + X
#     model.eval()
    with torch.no_grad():
        yp = model(data)
#     predictions = []
#     with torch.no_grad():
#         for xx in data:
#             predictions.append(model(xx.view(1, 3, 32, 32)).view(1, -1))
#     yp = torch.stack(predictions, dim=0).squeeze()
        Zi = nn.CrossEntropyLoss(reduction="none")(yp, y.repeat(yp.shape[0])).detach().cpu().numpy()
#     Zi = nn.CrossEntropyLoss(reduction="none")(yp, y.repeat(yp.shape[0])).cpu().numpy()
    Zi = Zi.reshape(*Xi.shape)
    #Zi = (Zi-Zi.min())/(Zi.max() - Zi.min())

    return Xi, Yi, Zi


def swa_schedule(epoch, lr_init=0.05, epochs=300):
    # adapted from https://github.com/wjmaddox/swa_gaussian/swag/run_swag.py
    t = epoch / epochs
    # t = (epoch) / (args.swa_start if args.swa else args.epochs)
    # lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    # lr_ratio = 0.02 / 0.01
    lr_ratio = 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    # return args.lr_init * factor
    return lr_init * factor


def jacobian(outputs, inputs, create_graph=False):
    jac = []
    flat_outputs = outputs.reshape(-1)
    grad_outputs = torch.zeros_like(flat_outputs)
    for i in range(len(flat_outputs)):
        grad_outputs[i] = 1.
        grad_inputs, = torch.autograd.grad(
            flat_outputs, inputs, grad_outputs, retain_graph=True, create_graph=create_graph
        )
        jac.append(grad_inputs.reshape(inputs.shape))
        grad_outputs[i] = 0.
    return torch.stack(jac).reshape(outputs.shape + inputs.shape)


def hessian(outputs, inputs):
    return jacobian(jacobian(outputs, inputs, create_graph=True), inputs)


def ntk(model, inp):
    """Calculate the neural tangent kernel of the model on the inputs.
    Returns the gradient feature map along with the tangent kernel.
    """
    out = model(inp)
    p_vec = nn.utils.parameters_to_vector(model.parameters())
    p, = p_vec.shape
    n, outdim = out.shape
    assert outdim == 1, "cant handle output dim higher than 1 for now"
    # this is the transpose jacobian (grad y(w))^T)
    features = torch.zeros(n, p, requires_grad=False)
    for i in range(n):  # for loop over data points
        model.zero_grad()
        out[i].backward(retain_graph=True)
        p_grad = torch.tensor([], requires_grad=False)
        for p in model.parameters():
            p_grad = torch.cat((p_grad, p.grad.reshape(-1)))
        features[i, :] = p_grad
    tk = features @ features.t()  # compute the tangent kernel
    return features, tk


def train(model, opt, data_loader, criterion, device, scheduler=None, polar=None, patches=False, transforms=None):
    n_samples, error, correct = 0.0, 0.0, 0.0
    model.train()
    for x, y in data_loader:
        if polar:
            x = polar_transform(x, transform_type='logpolar')
        if patches:
            max_patches = 10
            x = create_patches(x, size=(24, 24), max_patches=10, transforms=transforms)
            x, y = x.to(device), torch.cat([y] * max_patches, dim=0).to(device)
        else:
            x, y = x.to(device), y.to(device)
        batch_size = len(x)
        logits = model(x)
        loss = criterion(logits, y)

        y_hat = F.softmax(logits, dim=1).argmax(dim=1)
        correct += y.eq(y_hat.view_as(y)).sum().item()
        error += batch_size * loss.item()
        n_samples += batch_size

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()

    avg_loss = error / n_samples
    avg_acc = correct / n_samples

    return avg_loss, avg_acc


def test(model, data_loader, criterion, device, ood=False, polar=None, patches=False, transforms=None, flag_eval=True):
    n_samples, error, correct = 0.0, 0.0, 0.0
    collect_logits = []
    margin = torch.Tensor([]).to(device)
    if flag_eval:
        model.eval()
    with torch.no_grad():
        for x, y in data_loader:
#             if data_loader returns tuple of images, 
#             the clean image is in position 0
            x = x[0] if isinstance(x, list) else x
            y = y[0] if isinstance(y, list) else y
#             if evaluating on OoD data then scale labels, e.g. [0-99] to [0-9]
            if ood:
                y = ((y / 100) * 10).floor().long()
            if polar:
                x = polar_transform(x, transform_type='logpolar')
            if patches:
                max_patches = 10
                x = create_patches(x, size=(24, 24), max_patches=10, transforms=transforms)
                x, y = x.to(device), torch.cat([y] * max_patches, dim=0).to(device)
            else:
                x, y = x.to(device), y.to(device)
            batch_size = len(x)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, y)  # sum up batch loss
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

            y_hat = F.softmax(logits, dim=1).argmax(dim=1) # get the index of the max log-probability
            correct += y.eq(y_hat.view_as(y)).sum().item()
            error += batch_size * loss.item()
            n_samples += batch_size

            collect_logits.append(logits)

             # compute the margin
            probs = F.softmax(logits, dim=1)
            probs_clone = probs.clone()
            min_prob = probs_clone.min(dim=1)[0]
            probs_clone = probs_clone.scatter_(1, y.view(-1, 1), min_prob.view(-1, 1))
#             for i in range(y.size(0)):
#                 probs_clone[i, y[i]] = probs_clone[i, :].min()
            margin = torch.cat((margin, probs[:, y].diag() - probs_clone[:, probs_clone.max(dim=1)[1]].diag()), dim=0)

        te_margin = np.percentile(margin.cpu().numpy(), 5)
        te_margin = np.nan_to_num(te_margin, nan=0.0, posinf=0.0, neginf=0.0)

    avg_loss = error / n_samples
    avg_acc = correct / n_samples

    return avg_loss, avg_acc, te_margin, collect_logits


def calc_margin(logits, y):
    margin = torch.Tensor([]).to(logits.device)
    # compute the margin
    probs = F.softmax(logits, dim=1)
    probs_clone = probs.clone()
    min_prob = probs_clone.min(dim=1)[0]
    probs_clone.scatter_(1, y.view(-1, 1), min_prob.view(-1, 1))
    margin = torch.cat([margin, probs[:, y].diag() - probs_clone[:, probs_clone.max(dim=1)[1]].diag()], dim=0)
    te_margin = torch.quantile(margin, 0.05)
    return te_margin


def save_checkpoint(state, is_best, best_accuracy, filename='./checkpoints/baseline.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best, best_valid_acc: {}".format(best_accuracy))
        torch.save(state, filename)  # save checkpoint
#     else:
#         print ("=> Validation Accuracy did not improve")


def scaled_dot_product(query, key, value, mask=None):
    d_k = query.size()[-1]
    attn_logits = torch.matmul(query, key.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, value)
    return values, attention


def create_checkerboard_mask(h, w, invert=False):
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y)
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask

def create_channel_mask(c_in, invert=False):
    mask = torch.cat([torch.ones(c_in//2, dtype=torch.float32),
                      torch.zeros(c_in-c_in//2, dtype=torch.float32)])
    mask = mask.view(1, c_in, 1, 1)
    if invert:
        mask = 1


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     worker_init_fn=seed_worker
# )

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def optimizer2device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def enable_dropout(m):
    if type(m) == torch.nn.Dropout and m.training == False:
        m.train()


def average_predictions(model, X, y, n_classes=3, n_models=30):
#     predictions = np.zeros((len(X), num_classes))
    logits = np.zeros((len(X), num_classes))
    targets = np.zeros(len(X))
#     print(targets.size)

    for i in range(num_models):
        print("%d/%d" % (i + 1, num_models))
        model.eval()
        model.apply(enable_dropout)

#         k = 0
#         for x, target in zip(X, y):
#             torch.manual_seed(i)

# #             output = model(input)
#             outputs = model(x)

#             with torch.no_grad():
# #                 predictions[k : k + input.size()[0]] += (
# #                     F.softmax(output, dim=1).cpu().numpy()
# #                 )
#                 logits[k : k + x.size()[0]] += outputs.cpu().numpy()
# #             targets[k : (k + target.size(0))] = target.numpy()
#             k += x.size()[0]

#         full batch
        torch.manual_seed(i)
        with torch.no_grad():
            outputs = model(X)
            logits += outputs.cpu().numpy()

        print("Accuracy:", np.mean(np.argmax(logits, axis=1) == targets))

#     predictions /= num_models
    logits /= num_models

#     entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
#     np.savez(args.save_path, entropies=entropies, predictions=predictions, targets=targets)
    return logits