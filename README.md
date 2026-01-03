# Federated Learning with Byzantine-Robust Aggregation (CPU + Joblib)

This repository implements a **federated learning framework** in PyTorch with support for **Byzantine (faulty or malicious) clients** and multiple **robust aggregation strategies**.
The system is intentionally designed for **CPU-based parallel client training** using `joblib`, ensuring correctness, reproducibility, and multiprocessing safety.

## 📁 Project Structure
.
├── main.py                  # Entry point (training loop)
├── data/
│   ├── train/               # Training client data (JSON)
│   └── test/                # Test client data (JSON)
└── federated/
    ├── config.py            # Experiment-level configuration
    ├── optim_config.py      # Optimizer definitions
    ├── fedbase.py           # Server and Clients classes
    ├── client.py            # Client-side local training logic
    ├── aggregation.py       # Aggregation algorithms
    ├── autogm.py            # Auto-GM implementation
    ├── model_definition.py  # CNN model architecture
    └── data_utils.py        # Dataset loading and caching
---

## 🔍 Key Features
* **Federated Learning (FL)** with server–client architecture
* **Byzantine fault simulation**
  * Data poisoning
  * Model corruption
  * Noise-based attacks
* **Robust aggregation algorithms**
  * FedAvg
  * Coordinate-wise Median
  * Trimmed Mean
  * Krum
  * FedMI
  * aGMaMI (Auto-GM + Mutual Information)
* **CPU-safe parallel client training** using `joblib`
* **Clean experiment configuration** via dataclasses
* **Reproducible experiments** (fixed seeds)
* FEMNIST-style CNN model (62 classes)
---

## ⚙️ Configuration
### Experiment Configuration
Global experiment parameters are defined in: federated/config.py
---
### Optimizers
Defined in: federated/optim_config.py
---

## 🧠 Design Choice: CPU + Joblib
This project intentionally avoids GPU-based multiprocessing because:
* CUDA contexts are **not fork-safe**
* `joblib` uses process-based parallelism
* CPU training guarantees:
  * correctness
  * reproducibility
  * stable execution
> 💡 GPU support can be added later using `torch.multiprocessing`
---

## 🧯 Attack Models Supported
Configured when creating clients in `main.py`:

| Attack Type | Description              |
| ----------- | ------------------------ |
| `data`      | Label corruption         |
| `model`     | Randomized model weights |
| `noise`     | Additive Gaussian noise  |

---

## 📊 Aggregation Methods
Implemented in: federated/aggregation.py
* **FedAvg**
* **Coordinate-wise Median**
* **Trimmed Mean**
* **Krum**
* **FedMI**
* **aGMaMI** (Auto-GM + MI-based scoring)
These methods are designed to tolerate Byzantine or malicious client updates.
---

## 📈 Output and Logging
During training, the system reports:
* communication round index
* number of Byzantine clients
* global model accuracy
* global model loss

The framework is easily extensible to:
* save checkpoints
* log metrics to disk
* generate plots
---

## 🔬 Intended Use
This codebase is intended for:
* federated learning research
* Byzantine-robust aggregation experiments
* academic projects, theses, or papers
* reproducible experimental studies

It is **not** intended as a production FL system.
---

## 📜 License
This project is intended for **research and educational use**.
---
