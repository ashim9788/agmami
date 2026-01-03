import torch
import numpy as np
from numpy import linalg as LA
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from federated.fedbase import Server
from federated.model_definition import simpleCNN
from collections import defaultdict

def fedavg(local_weight_list):
    # Transpose the list of updates to group weights by layer
    # From shape [num_clients][num_layers] to [num_layers][num_clients]
    grouped_updates = list(zip(*local_weight_list))
    # Compute the mean for each layer
    aggregated_weights = [np.mean(np.stack(layer_updates), axis=0) for layer_updates in grouped_updates]
    return aggregated_weights

def krum(local_weight_list, num_byzantine):
    n = len(local_weight_list)
    scores = []
    flattened_weights = [np.concatenate([w.flatten() for w in weights]) for weights in local_weight_list]
    for i, update_i in enumerate(flattened_weights):
        distances = []
        # Compute the distance between update_i and all other updates
        for j, update_j in enumerate(flattened_weights):
            if i != j:
                distance = np.linalg.norm(update_i - update_j)
                distances.append(distance)
        # Sort distances and sum the smallest n - f - 2 distances
        distances = np.sort(distances)
        score = np.sum(distances[:n - num_byzantine - 2])
        scores.append(score)
    # Select the index with the minimum score
    selected_index = np.argmin(scores)
    # Return the corresponding local weights as the aggregated result
    return local_weight_list[selected_index]

def trimmed_mean(local_weight_list, num_byzantine):
    n_clients = len(local_weight_list)
    n_layers = len(local_weight_list[0])
    trimmed_weights = []
    for layer_index in range(n_layers):
        layer_weights = np.array([client_weights[layer_index] for client_weights in local_weight_list])
        # Check if the layer_weights array is empty
        if layer_weights.size == 0:
            continue
        trimmed_layer = np.zeros_like(layer_weights[0])
        for weight_index in np.ndindex(layer_weights[0].shape):
            # Access weight values for the current index across all clients
            weight_values = layer_weights[(slice(None),) + weight_index]

            # Sort the values and trim the outliers
            weight_values_sorted = np.sort(weight_values)
            trimmed_values = weight_values_sorted[num_byzantine: n_clients - num_byzantine]
            # Compute the mean of the remaining (trimmed) values
            trimmed_layer[weight_index] = np.mean(trimmed_values)
        # Append the trimmed mean of the current layer to the result
        trimmed_weights.append(trimmed_layer)
    return trimmed_weights

def coordinate_median(local_weight_list):
    n_layers = len(local_weight_list[0])  # Number of layers in the model
    median_weights = []
    for layer_index in range(n_layers):
        # Stack the weights for the current layer from all clients
        layer_weights = np.array([client_weights[layer_index] for client_weights in local_weight_list])
        # Initialize an array to hold the coordinate-wise median of the current layer
        median_layer = np.median(layer_weights, axis=0)
        # Append the coordinate-wise median of the current layer to the result
        median_weights.append(median_layer)
    return median_weights

def calculate_outputs(update, global_weights, dataset, model_architecture):
    device = torch.device('cpu')
    # Build the model and compute outputs for the given update
    node_model = model_architecture().to(device)
    local_weights = [torch.tensor(g) + torch.tensor(u) for g, u in zip(global_weights, update)]
    # Load weights into the PyTorch model
    state_dict = node_model.state_dict()
    new_state_dict = {k: local_weights[i] for i, k in enumerate(state_dict.keys())}
    node_model.load_state_dict(new_state_dict)

    # node1_outputs = node1_model(dataset)
    # Convert dataset to tensor and move to device
    if isinstance(dataset, (np.ndarray, np.memmap)):
        dataset = torch.tensor(dataset).permute(0, 3, 1, 2).float()
    #dataset = dataset.to(device)
    assert torch.isfinite(dataset).all(),"Input contains NaNs or Infs"

    node_model.eval()
    with torch.no_grad():
        node_outputs = node_model(dataset)
    return node_outputs # .cpu().numpy()

def calculate_mi(output1, output2):
    """
    Args:
        output1: Tensor of shape [batch_size, num_features] (on GPU)
        output2: Tensor of shape [batch_size, num_features] (on GPU)   
    Returns:
        Scalar tensor: average mutual information over the batch
    """
    # Ensure tensors are on the same device
    assert output1.shape == output2.shape, "Outputs must have the same shape"
    device = output1.device
    # Flatten if needed
    output1 = output1.view(output1.size(0), -1)
    output2 = output2.view(output2.size(0), -1)
    # Normalize to zero mean, unit variance
    eps = 1e-10  # for numerical stability
    output1 = (output1 - output1.mean(dim=1, keepdim=True)) / (output1.std(dim=1, keepdim=True) + eps)
    output2 = (output2 - output2.mean(dim=1, keepdim=True)) / (output2.std(dim=1, keepdim=True) + eps)
    # Compute Pearson correlation for each sample
    rho = (output1 * output2).mean(dim=1)
    rho = torch.clamp(rho, -1 + eps, 1 - eps)
    # Mutual Information: MI = -0.5 * log(1 - rho^2)
    mi = -0.5 * torch.log(1 - rho ** 2 + eps)
    return mi.mean()  # Average MI over batch

def compute_score_for_index_fedmi(current_output, outputs):
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_output = current_output.to(device)
    score = torch.tensor(0.0, device=device)
    # score = 0.0
    count = 0
    for other_output in outputs:
        other_output = other_output.to(device)
        if not torch.equal(current_output, other_output):
            score += calculate_mi(current_output, other_output)
            count += 1
    mi_score = score / count if count > 0 else 0.0
    return mi_score.cpu().numpy()

def mad_based_filter(mi_values):
    median = np.median(mi_values)
    madn = np.median(np.abs(mi_values - median))
    # madn = np.median(np.abs(mi_values - median))/0.6745
    lower_bound = median - 2 * madn
    upper_bound = median + 2 * madn
    return [i for i, mi in enumerate(mi_values) if lower_bound <= mi <= upper_bound]

def fedmi(updates, global_weights, dataset, model_architecture, n_jobs):
    # Compute the outputs of all models on the dataset
    outputs = Parallel(n_jobs=n_jobs)(delayed(calculate_outputs)(update, global_weights, dataset, model_architecture)
        for update in updates)
    # Compute the MI scores for each model
    scores = Parallel(n_jobs=n_jobs)(delayed(compute_score_for_index_fedmi)(client_output, outputs)
        for client_output in outputs)
    mi_scores = [float(x) for x in scores]
    # print("MI scores: ", mi_scores)
    # Select the model with the highest score
    valid_indices = mad_based_filter(mi_scores)
    print("Valid Indices (FedMi): ", len(valid_indices))
    valid_updates = [updates[i] for i in valid_indices]
    if len(valid_updates) == 0:
        return None
    return fedavg(valid_updates)  


# agmami helper functions
def aggregate(update_list, alphas=None):
    weights = alphas
    norm_weights = np.array(weights) / np.sum(weights)
    avg_updates = np.sum([param * weight for param, weight in zip(update_list, norm_weights)])
    
    return avg_updates

def l2dist(model1, model2):
    return LA.norm(model1 - model2)

def geometric_median_objective(median, points, alphas):
    return sum([alpha * l2dist(median, p) for alpha, p in zip(alphas, points)])

def auto_gm_agmami(mi_scores, lamb, maxiter=100, eps=1e-5, ftol=1e-6):

    distance = []
    lamb = lamb * (len(mi_scores))
    alpha = np.ones(shape=len(mi_scores)) / len(mi_scores)
    # alpha is an array of dimensions 1xnum_clients with each element=1/numclients
    global_model = None
    for i in range(maxiter):
        median = aggregate(mi_scores, alpha)
        obj_val = geometric_median_objective(median, mi_scores, alpha)
        global_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
        for j in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray(
                [alpha / max(eps, l2dist(median, p)) for alpha, p in zip(alpha, mi_scores)],
                dtype=alpha.dtype)
            weights = weights / weights.sum()
            median = aggregate(mi_scores, weights)
            obj_val = geometric_median_objective(median, mi_scores, alpha)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        
        global_model = median
        for mi in mi_scores:
            distance.append(l2dist(median, mi))
        
        # Update weights
        idxs = [x for x, _ in sorted(enumerate(distance), key=lambda x: x[1])]
        # idxs = [x for x, _ in sorted(enumerate(local_clients_list), key=lambda x: x[1].distance)]
        eta_optimal = distance[idxs[0]] + lamb
        for p in range(0, len(idxs)):
            eta = (sum([distance[ii] for ii in idxs[:p + 1]]) + lamb) / (p + 1)
            if p < len(idxs) and eta - distance[idxs[p]] < 0:
                break
            else:
                eta_optimal = eta
        alpha = np.array([max(eta_optimal - distance[c], 0) / lamb for c in range(len(mi_scores))])
        
        new_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
        if abs(new_obj - global_obj) < ftol * new_obj:
            break

    return alpha, global_model

def agmami(updates, global_weights, X_test, model_architecture, lamb, n_jobs):
    # Compute the outputs of all models on the dataset
    outputs = Parallel(n_jobs=n_jobs)(delayed(calculate_outputs)(update, global_weights, X_test, model_architecture)
        for update in updates)
    # print(outputs[0].device)
    scores = Parallel(n_jobs=n_jobs)(delayed(compute_score_for_index_fedmi)(client_output, outputs)
        for client_output in outputs)
    mi_scores = [float(x) for x in scores]
    print("MI scores: ", mi_scores)
    # For the new code, calculate the auto-GM of the mi scores by reweighting and regularization;
    # and determine the valid_indices
    alphas, gm = auto_gm_agmami(mi_scores, lamb)
    print("alpha values: ", alphas)
    nonzero_indices = np.nonzero(alphas)[0]
    # nonzero_indices = [i for i, val in enumerate(alphas) if val > 0] # autogm based filter
    if len(nonzero_indices) == 0:
        print("No non-zero Indices")
        return None
    print(f"Valid Indices (aGMaMI): {len(nonzero_indices)}")
    return fedavg([updates[i] for i in nonzero_indices])
