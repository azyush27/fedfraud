import logging
from flwr.simulation import run_simulation
from client import app as client_app, N_CLIENTS
from server import app as server_app

logging.getLogger("flwr").setLevel(logging.INFO)

if __name__ == "__main__":
    print(f"Starting federated simulation with {N_CLIENTS} simulated banks...\n")
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=N_CLIENTS,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )