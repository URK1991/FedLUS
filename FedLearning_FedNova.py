import flwr as fl
from client_FedAvg import LUSClient
from flwr.server.strategy import FedAvg
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import numpy as np
import torch.nn as nn
from models import get_model
from functools import reduce


from flwr.server.strategy.aggregate import aggregate
from flwr.common import (
    Metrics,
    NDArray,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from logging import WARNING

#Selecting the Model Architecture

model = get_model_R18_SA(num_classes=4).to(device)
    
#defining client IDs
def client_fn(cid: str):
    centers = {
        '0': "Brescia",
        '1': "Lucca",
        '2': "Pavia",
        '3': "Rome",
        '4': "Tione",
    }
    # Return a standard Flower client
    return LUSClient(centers[cid])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#logging the information in the file named as log.txt
fl.common.logger.configure(identifier="LUS_FL_Experiment", filename="log.txt")

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None

if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

class FedNova(fl.server.strategy.FedAvg):
     def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict[str, fl.common.Scalar]]:
        if not results:
            return None, {}

        total_weight = 0.0
        weighted_updates = None
        
        local_tau = [res.metrics['num_iterations'] for _, res in results]
        tau_eff = np.sum(local_tau)
        tau_eff = tau_eff/5 #5 represents the number of medical centers
        tot_examples = [f_res.num_examples for _, f_res in results]
        tot_examples = np.sum(tot_examples)
        
        for client, fit_res in results:
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            num_iterations = fit_res.metrics['num_iterations']
            
            ratio = num_examples/tot_examples
            weight = tau_eff/num_iterations
            weight = weight*ratio

            if weighted_updates is None:
                weighted_updates = [weight * nu for nu in weights]
            else:
                for i in range(len(weighted_updates)):
                    weighted_updates[i] += weight * weights[i]

        #aggregated_updates = [wu / total_weight for wu in weighted_updates]
        aggregated_parameters = fl.common.ndarrays_to_parameters(weighted_updates)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, {}


# To implement the strategy
strategy = FedNova()

print(client_resources)
#Launch the simulation
hist = fl.simulation.start_simulation(
    client_fn=client_fn, # A function to run a _virtual_ client when required
    num_clients=5, # Total number of clients available
    config=fl.server.ServerConfig(num_rounds=250), # Specify number of FL rounds
    client_resources=client_resources,
    strategy=strategy # A Flower strategy
)
