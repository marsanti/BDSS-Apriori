import argparse
import yaml
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

@dataclass
class ArgsType:
    seed: int
    verbosity: str
    frontier_only: bool
    csv_path: str
    out_dir: str
    relation_file: Optional[str]
    rules_file: Optional[str]
    n_cols: int
    n_rows: int
    missing_value_key: int
    max_bins: int
    alg: str
    eps: float
    min_conf: float
    n_permutations: int
    max_p_value: float
    min_lift: float
    max_iter: int
    n_workers: int

class Config:
    def __init__(self):
        # Parser
        self.parser = argparse.ArgumentParser(description='Frequent Itemsets Extraction Configurations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self._add_args()

        # Arguments
        namespace_args = self.parser.parse_args()
        self._args: ArgsType = ArgsType(
            seed=namespace_args.seed,
            verbosity=namespace_args.verbosity,
            frontier_only=namespace_args.frontier_only,
            csv_path=namespace_args.csv_path,
            out_dir=namespace_args.out_dir,
            relation_file=namespace_args.relation_file,
            rules_file=namespace_args.rules_file,
            n_cols=namespace_args.n_cols,
            n_rows=namespace_args.n_rows,
            missing_value_key=namespace_args.missing_value_key,
            max_bins=namespace_args.max_bins,
            alg=namespace_args.alg,
            eps=namespace_args.eps,
            min_conf=namespace_args.min_conf,
            n_permutations=namespace_args.n_permutations,
            max_p_value=namespace_args.max_p_value,
            min_lift=namespace_args.min_lift,
            max_iter=namespace_args.max_iter,
            n_workers=namespace_args.n_workers,
        )

        # output directory setup
        csv_path = self._args.csv_path
        now = int(datetime.now().timestamp())
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        self._out_dir = os.path.join(self._args.out_dir, dataset_name, f'{self._args.alg}_{now}')
        os.makedirs(self._out_dir, exist_ok=True)

        self._save_yaml()
    
    def _save_yaml(self):
        with open(f'{self._out_dir}/config.yaml', 'w') as f:
            yaml.dump(vars(self._args), f)

    def _add_args(self):
        # general settings
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        self.parser.add_argument('--verbosity', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging verbosity level')
        self.parser.add_argument('--frontier_only', action='store_true', help='If set, only the frontier itemsets are extracted (Recommended to avoid relation file size explosion)')
        # files settings
        self.parser.add_argument('--csv_path', type=str, default='datasets/AirQualityUCI.csv', help='Path to the CSV dataset')
        self.parser.add_argument('--out_dir', type=str, default='results/', help='Output directory for results')
        self.parser.add_argument('--relation_file', type=str, default=None, help='Path to the pickle relation file')
        self.parser.add_argument('--rules_file', type=str, default=None, help='Path to the pickle rules file')
        # dataset settings
        self.parser.add_argument('--n_cols', type=int, default=0, help='Number of columns to sample (0=all)')
        self.parser.add_argument('--n_rows', type=int, default=0, help='Number of rows to sample (0=all)')
        self.parser.add_argument('--missing_value_key', type=int, default=-200, help='Key representing missing values in the dataset')
        self.parser.add_argument('--max_bins', type=int, default=15, help='Maximum number of bins for quantitative attributes: it controls the discretization granularity')
        # Standard Apriori settings
        # Frequent Itemsets extraction settings
        self.parser.add_argument('--alg', type=str, default='apriori', choices=['apriori', 'randomic', 'distributed'], help='Extraction algorithm')
        self.parser.add_argument('--eps', type=float, default=0.8, help='Support threshold')
        # Extaction rules settings
        self.parser.add_argument('--min_conf', type=float, default=0.8, help='Minimum confidence for rule extraction')
        self.parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for Rule Extraction and Distributed Apriori')
        # Shapley settings
        self.parser.add_argument('--max_p_value', type=float, default=0.05, help='Maximum p-value threshold for filtering rules in Shapley analysis')
        self.parser.add_argument('--min_lift', type=float, default=1.5, help='Minimum lift threshold for filtering rules in Shapley analysis')
        self.parser.add_argument('--n_permutations', type=int, default=50, help='Number of permutations for Shapley value estimation')
        # Randomic Apriori settings
        self.parser.add_argument('--max_iter', type=int, default=0, help='Maximum number of iterations for Randomic Apriori, if 0 no limit')

    @property
    def args(self) -> ArgsType:
        return self._args
    
    @property
    def out_dir(self) -> str:
        return self._out_dir
