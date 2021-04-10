import argparse
import torch
from glob import glob
import numpy as np


def load_and_aggregate_agent_params(file_path: str) -> np.ndarray:
    agent = torch.load(file_path, map_location=torch.device('cpu'))[0]
    param_vector = np.array([])
    for param in agent.values():
        param_vector = np.append(param_vector, param.detach().numpy().ravel())
    return param_vector.reshape(-1, 1)


def _main(args: argparse.Namespace):
    files = glob(args.base_path)
    param_vectors = [load_and_aggregate_agent_params(f) for f in files]
    all_params = np.hstack(param_vectors)
    relative_variance = all_params.std(axis=1) / np.abs(all_params).mean(axis=1)
    print(np.abs(relative_variance).mean())
    return relative_variance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str,
                        default="~/code/DiffDAC-Experiments/independent/RandomWindHopper/trained_models/final_params_*.pt",
                        help="Generic file path for glob to collect all param files.")
    return parser.parse_args()


if __name__ == "__main__":
    _main(parse_args())
