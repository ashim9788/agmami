import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cdist, euclidean

class AutogmClient:
    def __init__(self, parameter, num_train_samples=1):
        self.parameter = parameter
        self.num_train_samples = num_train_samples
        self.distance = None

def get_model_list(local_weight_list, global_model):
    model_list = []
    for local_weight in local_weight_list:
        local_model = [l + g for l, g in zip(local_weight, global_model)]
        model_list.append(local_model)
    return model_list

def flatten_and_concatenate(weights_list):
    """
    Flatten and concatenate a list of NumPy arrays into a single 1D NumPy array.
    """
    flattened_weights = [w.flatten() for w in weights_list]
    concatenated_weights = np.concatenate(flattened_weights)
    return concatenated_weights

def autogm_preprocessing(model_updates):
    clients = []
    for weights in model_updates:
        # Flatten and concatenate the weights
        parameter = flatten_and_concatenate(weights)
        # Create a Client object
        client = AutogmClient(parameter=parameter, num_train_samples=1)
        clients.append(client)
    return clients

def reconstruct_weights(flattened_weights, original_weight_shapes):
    """
    Reconstruct the list of weights from the flattened array using the original shapes.
    """
    weights = []
    index = 0
    for shape in original_weight_shapes:
        size = int(np.prod(shape))
        layer_weights = flattened_weights[index:index + size].reshape(shape)
        weights.append(layer_weights)
        index += size
    return weights

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)
    
    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]
        
        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)
        
        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y
        
        if euclidean(y, y1) < eps:
            return y1
        
        y = y1

def aggregate(update_list, alphas=None):
    weights = alphas
    nor_weights = np.array(weights) / np.sum(weights)
    avg_updates = np.sum(np.stack([param * weight for param, weight in zip(update_list, nor_weights)]), axis=0) 
    return avg_updates

def l2dist(model1, model2):
    return LA.norm(model1 - model2)

def geometric_median_objective(median, points, alphas):
    return sum([alpha * l2dist(median, p) for alpha, p in zip(alphas, points)])

def weighted_average_oracle(points, weights):
    tot_weights = np.sum(weights)
    weighted_updates = [np.zeros_like(v) for v in points[0]]
    for w, p in zip(weights, points):
        for j, weighted_val in enumerate(weighted_updates):
            weighted_val += (w / tot_weights) * p[j]
    return weighted_updates

def auto_gm(local_weight_list, global_weights, lamb=1.0, maxiter=100, eps=1e-5, ftol=1e-6):
    temp = get_model_list(local_weight_list, global_weights)
    local_clients_list = autogm_preprocessing(temp) # here local_clients_list is a list of client objects that has weights, distance and number of training samples as parametrs
    param_list = [client.parameter for client in local_clients_list]
    lamb = lamb * (len(local_clients_list))
    alpha = np.ones(shape=len(local_clients_list)) / len(local_clients_list)
    global_model_flattened = None
    for i in range(maxiter):
        median = aggregate(param_list, alpha)
        obj_val = geometric_median_objective(median, param_list, alpha)
        global_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
        for j in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray(
                [alpha / max(eps, l2dist(median, p)) for alpha, p in zip(alpha, param_list)],
                dtype=alpha.dtype)
            weights = weights / weights.sum()
            median = aggregate(param_list, weights)
            obj_val = geometric_median_objective(median, param_list, alpha)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        global_model_flattened = median
        # print("Flattened weights' shape: ", np.shape(global_model_flattened))
        for client in local_clients_list:
            client.distance = l2dist(median, client.parameter)
        # Update weights
        idxs = [x for x, _ in sorted(enumerate(local_clients_list), key=lambda x: x[1].distance)]
        eta_optimal = local_clients_list[idxs[0]].distance + lamb / local_clients_list[idxs[0]].num_train_samples
        for p in range(0, len(idxs)):
            eta = (sum([local_clients_list[ii].distance for ii in idxs[:p + 1]]) + lamb) / (p + 1)
            if p < len(idxs) and eta - local_clients_list[idxs[p]].distance < 0:
                break
            else:
                eta_optimal = eta
        alpha = np.array([max(eta_optimal - c.distance, 0) / lamb for c in
                          local_clients_list])
        new_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
        if abs(new_obj - global_obj) < ftol * new_obj:
            break
        original_weight_shapes = [w.shape for w in global_weights]
        global_model = reconstruct_weights(global_model_flattened, original_weight_shapes)
    nonzero_indices = np.nonzero(alpha)[0]
    print(f"Valid Indices (autoGM): {len(nonzero_indices)}")
    return alpha, global_model