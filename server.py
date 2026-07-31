from __future__ import annotations

from typing import Any

from flwr.common import FitRes, Parameters
from flwr.server import ClientManager, Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.serverapp import ServerApp


class FedAvgWithMetrics(FedAvg):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.round_metrics: dict[int, dict[str, float]] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, BaseException] | None],
    ) -> tuple[Parameters, dict[str, Any]]:
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if results:
            sample_counts = [res.num_examples for _, res in results if res.num_examples is not None]
            total_samples = sum(sample_counts) if sample_counts else 0
            weighted_metrics = {}
            for metric_name in {metric for _, res in results for metric in res.metrics.keys()}:
                weighted_sum = sum(
                    res.metrics.get(metric_name, 0.0) * res.num_examples
                    for _, res in results
                    if res.num_examples is not None and metric_name in res.metrics
                )
                weighted_metrics[metric_name] = weighted_sum / total_samples if total_samples else 0.0
            aggregated_metrics = {**(aggregated_metrics or {}), **weighted_metrics}
        self.round_metrics[server_round] = aggregated_metrics or {}
        return aggregated_params, aggregated_metrics


def make_server_app() -> ServerApp:
    strategy = FedAvgWithMetrics(min_fit_clients=2, fraction_fit=1.0)
    return ServerApp(strategy=strategy)
