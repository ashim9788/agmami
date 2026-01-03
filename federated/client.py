import torch
import random
import numpy as np

# Helper: build model and set weights
def _set_model_weights(model, weights):
    state_dict = model.state_dict()
    new_state_dict = {}
    for (key, param), w in zip(state_dict.items(), weights):
        new_state_dict[key] = torch.as_tensor(
            w,
            dtype=param.dtype,
            device=param.device
        )
    model.load_state_dict(new_state_dict, strict=True)

def _get_model_weights(model):
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]

def generate_random_state_dict(model, mean, std):
    random_dict = {}
    for k, v in model.state_dict().items():
        if torch.is_floating_point(v):
            rand_tensor = torch.normal(mean=torch.full_like(v, mean),
                                       std=torch.full_like(v, std))
        else:
            rand_tensor = v.clone()  # keep original for int/bool buffers
        random_dict[k] = rand_tensor.cpu().numpy()
    return random_dict
    # The rand_tensor works fine for floating-point tensors (e.g., float32), 
    # but will fail for integer buffers in state_dict() 
    # (like num_batches_tracked in BatchNorm).

def _train_client_function(global_weights, local_data, local_labels, faulty, model_arch, 
                           loss_fn, optimizer_cls, optimizer_args, metrics, attack_type, local_epochs=10):
    if len(local_data.shape) == 3:
        # (batch_size, 28, 28) → (batch_size, 1, 28, 28)
        local_data = local_data[:, np.newaxis, :, :]
    elif local_data.shape[1] != 1:
        raise ValueError(f"Expected input with 1 channel, got shape: {local_data.shape}")
    device = torch.device("cpu")
    model = model_arch().to(device)
    _set_model_weights(model, global_weights)
    optim = optimizer_cls(model.parameters(), **optimizer_args)
    if not faulty:
        dataset = torch.utils.data.TensorDataset(torch.tensor(local_data, dtype=torch.float32),
                                                 torch.tensor(local_labels, dtype=torch.long))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
        model.train()
        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optim.step()
        local_weight = _get_model_weights(model)
    else:
        if attack_type == 'model':
            # local_weight = generate_random_state_dict(model, mean=0.0, std=random.uniform(0, 0.5))
            # OR you can use the following 
            mean = 0
            std = random.uniform(0, 0.5)
            # std = 0.1
            random_weights = []
            for v in model.state_dict().values():
                rand_np = np.random.normal(loc=mean, scale=std, size=v.shape)
                random_weights.append(rand_np.astype(np.float32))  # or match dtype
            local_weight = random_weights
        elif attack_type == 'noise':
            noise_scale = random.uniform(0, 0.1)
            poisoned_local_weight = []
            for w in global_weights:
                # Generate random noise with the same shape as the weight tensor
                noise = np.random.normal(loc=noise_scale, scale=noise_scale, size=w.shape)
                # Add the noise to the global weight
                noisy_weight = w + noise
                poisoned_local_weight.append(noisy_weight)
            local_weight = poisoned_local_weight
        elif attack_type == 'data':
            corrupt_labels = np.array([61 - l for l in local_labels])  # arbitrary corruption
            dataset = torch.utils.data.TensorDataset(torch.tensor(local_data, dtype=torch.float32),
                                                     torch.tensor(corrupt_labels, dtype=torch.long))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
            model.train()
            for _ in range(local_epochs):
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    optim.zero_grad()
                    output = model(x)
                    loss = loss_fn(output, y)
                    loss.backward()
                    optim.step()
            local_weight = _get_model_weights(model)
    return [l - g for l, g in zip(local_weight, global_weights)]
