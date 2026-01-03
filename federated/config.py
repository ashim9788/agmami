from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 3597
    device: str = "cpu"

    comm_rounds: int = 200
    clients_per_round: int = 10
    local_epochs: int = 10

    p_faulty: float = 0.2
    lamb: float = 0.3
