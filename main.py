import random
import numpy as np
import torch
from joblib import Parallel, delayed

from federated.fedbase import Clients, Server
from federated.model_definition import simpleCNN
from federated.data_utils import read_data
from federated.client import _train_client_function
from federated.aggregation import agmami
from federated.config import ExperimentConfig
from federated.optim_config import OPTIMIZERS

cfg = ExperimentConfig()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer_cls, optimizer_args = OPTIMIZERS["sgd"]

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

# -----------------------------
# Load data
# -----------------------------
users, train_data, test_data = read_data("data/train", "data/test")

# -----------------------------
# Server & Clients
# -----------------------------
server = Server(model_architecture=simpleCNN, device=cfg.device)

clients = Clients(
    model_architecture=simpleCNN,
    attack_type="data"
)
clients.create_clients(users, train_data, p_faulty=cfg.p_faulty)
client_ids = list(clients.get_clients().keys())

# -----------------------------
# Build global test set
# -----------------------------
def build_global_testset(users, test_data, client_ids, n_users, rng):
    X, y = [], []
    sampled = rng.sample(client_ids, n_users)
    for cid in sampled:
        uid = users[cid - 1]
        X.extend(test_data[uid]["x"])
        y.extend(test_data[uid]["y"])
    X = np.array(X).reshape(-1, 28, 28, 1)
    y = np.array(y)
    return X, y
_rng = random.Random(cfg.seed)
X_test, y_test = build_global_testset(users, test_data, client_ids, n_users=30, rng=_rng)

def main():
    print(
    f"ExperimentConfig: "
    f"rounds={cfg.comm_rounds}, "
    f"clients/round={cfg.clients_per_round}, "
    f"local_epochs={cfg.local_epochs}, "
    f"p_faulty={cfg.p_faulty}, "
    f"lamb={cfg.lamb}")
    # -----------------------------
    # Training loop
    # -----------------------------
    acc_history, loss_history = [], []

    for rnd in range(cfg.comm_rounds):
        print(f"\n--- Round {rnd + 1} ---")

        selected = random.sample(client_ids, cfg.clients_per_round)
        global_weights = server.get_weights()

        parallel_args = []
        byzantine = 0

        for cid in selected:
            info = clients.clients[cid]
            byzantine += int(info["faulty"])

            parallel_args.append((
                global_weights,
                info["data"],
                info["labels"],
                info["faulty"],
                simpleCNN,
                loss_fn,
                optimizer_cls,
                optimizer_args,
                ["accuracy"],
                clients.attack,
                cfg.local_epochs
            ))

        print(f"Byzantine clients: {byzantine}")

        # CPU-safe parallelism
        updates = Parallel(n_jobs=cfg.clients_per_round, backend="loky")(
            delayed(_train_client_function)(*args)
            for args in parallel_args
        )

        chosen_update = agmami(updates,global_weights,X_test,simpleCNN,cfg.lamb,n_jobs=cfg.clients_per_round)

        if chosen_update is not None:
            server.set_weights_others(chosen_update)

        acc, loss = server.test(X_test, y_test)
        acc_history.append(acc)
        loss_history.append(loss)

        print(f"Acc: {acc:.3%} | Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
