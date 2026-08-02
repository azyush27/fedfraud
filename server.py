from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, Context

from model import build_model, get_params

NUM_ROUNDS = 15


def fit_config(server_round: int):
    return {"local_epochs": 2}


def weighted_metrics_avg(results):
    total = sum(n for n, _ in results)
    if total == 0:
        return {}
    agg = {}
    for key in ("precision", "recall", "f1", "auc"):
        agg[key] = sum(n * m.get(key, 0.0) for n, m in results) / total
    return agg


def server_fn(context: Context):
    initial_params = ndarrays_to_parameters(get_params(build_model()))
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_params,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_metrics_avg,
        evaluate_metrics_aggregation_fn=weighted_metrics_avg,
    )
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)