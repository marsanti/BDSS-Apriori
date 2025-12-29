import pandas as pd
import pickle
import os
import time
from typing import Set

from config import Config
from src.logger import Logger
from src.dataset import Dataset
from src.itemset import Itemset
from src.algorithms import standard_apriori
from src.rule import RuleExtractor
from src.utils import show_rules, format_time
from src.shapley import ShapleyAnalyzer

def preprocessDataset(df: pd.DataFrame, config_args: dict, log: Logger) -> pd.DataFrame:
    """
    Cleans the dataset by removing metadata columns and handling missing values.
    Missing values are identified by the specified key (default -200).
    """
    # Drop metadata columns if they exist
    cols_to_drop = ['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Replace missing value keys with NaN
    df = df.replace(config_args.missing_value_key, pd.NA)
    
    # Drop rows with any NaN values
    df = df.dropna()

    # THIS IS NEEDED FOR TESTING PURPOSES ONLY, use n_cols and n_rows equal to 0 to disable
    # Column Sampling
    if config_args.n_cols > 0 and config_args.n_cols < len(df.columns):
        df = df.sample(n=config_args.n_cols, axis='columns', random_state=config_args.seed)

    log.debug(f"column selected: {df.columns}")

    # Row Sampling
    if config_args.n_rows > 0:
        # Safety check
        n = min(config_args.n_rows, len(df))
        df = df.sample(n=n, axis='index', random_state=config_args.seed)
    # log.debug(f"Indexes after sampling: {df.index.tolist()}")
    return df

def main():
    start_script_time = time.time()
    config = Config()
    log_path = os.path.join(config.out_dir, 'execution.log')
    log = Logger(__name__, log_file=log_path, level=config.args.verbosity)
    relation: Set[Itemset] = None 
    csv_path = config.args.csv_path

    try:
        # Step 1. Load Dataset
        log.info(f"Loading {csv_path}...")
        try:
            # Load with correct separator and decimal
            df = pd.read_csv(csv_path, sep=';', decimal=',')
        except FileNotFoundError:
            log.error("Error: 'AirQualityUCI.csv' not found. Please ensure the file is in the directory.")
            return

        original_len = len(df)
        df_cleaned = preprocessDataset(df, config.args, log)
        log.debug(f"Cleaned Dataset: Dropped {original_len - len(df_cleaned)} rows with missing values.")
        log.info(f"Dataset: {len(df_cleaned)} rows, {len(df_cleaned.columns)} attributes.")

        dataset: Dataset = Dataset(df_cleaned, config.args.max_bins)

        # Step 2. Frequent Itemsets Extraction
        # check if relation file exists
        if config.args.relation_file is None:
            log.info(f"Starting Frequent Itemsets Extraction, algorithm: {config.args.alg}...")
            start_extraction_time = time.time()

            match config.args.alg:
                case 'apriori':
                    dataset.R = standard_apriori(dataset, config.args.eps, config.args.frontier_only, log)
                case 'randomic_apriori':
                    dataset.R = {} # TODO
                case 'distributed_apriori':
                    dataset.R = {} # TODO
                case _:
                    log.error(f"Error: Unknown algorithm '{config.args.alg}' specified.")

            relation = dataset.R

            # save the relation in order to load without recomputing
            log.info(f"Saving relation to {config.out_dir}/relation.pkl...")
            with open(f'{config.out_dir}/relation.pkl', 'wb') as f:
                    pickle.dump(relation, f)

            end_extraction_time = time.time()
            log.info(f"Frequent Itemsets Extraction completed in {format_time(end_extraction_time - start_extraction_time)}.")
        else:
            # relation file is given
            if(not os.path.exists(config.args.relation_file)):
                log.error(f"Error: Specified relation file '{config.args.relation_file}' does not exist.")
                return
            log.info(f"Loading relation from {config.args.relation_file}...")
            with open(config.args.relation_file, 'rb') as f:
                relation = pickle.load(f)

        log.info(f"Total Itemsets extracted: {len(relation)}")

        if config.args.rules_file is None:
            log.info("Starting Rule Extraction...")
            start_rule_time = time.time()
            
            # Step 3. Extract Rules
            extractor = RuleExtractor(dataset, min_conf=config.args.min_conf, log=log)
            rules = extractor.extract_rules(relation)
            
            # Sort by Lift descending
            rules.sort(key=lambda x: x.lift, reverse=True)

            show_rules(rules, dataset, log, n_rules=10)

            # Save rules
            with open(f'{config.out_dir}/rules.pkl', 'wb') as f:
                pickle.dump(rules, f)

            end_rule_time = time.time()
            log.info(f"Rule Extraction completed in {format_time(end_rule_time - start_rule_time)}s. Total rules extracted: {len(rules)}")
        else:
            # rules file is given
            if(not os.path.exists(config.args.rules_file)):
                log.error(f"Error: Specified rules file '{config.args.rules_file}' does not exist.")
                return
            log.info(f"Loading rules from {config.args.rules_file}...")
            with open(config.args.rules_file, 'rb') as f:
                rules = pickle.load(f)
            log.info(f"Total Rules loaded: {len(rules)}")

            show_rules(rules, dataset, log, n_rules=10)

        log.info("Starting Shapley Analysis...")
        start_shapley_time = time.time()

        shapley = ShapleyAnalyzer(dataset, rules, log=log, max_p_value=config.args.max_p_value, min_lift=config.args.min_lift)
        
        # Execute Analysis
        shapley.execute(n_permutations=config.args.n_permutations)

        end_shapley_time = time.time()
        log.info(f"Shapley analysis Finished in {format_time(end_shapley_time - start_shapley_time)}s.")

        end_script_time = time.time()
        log.info(f"Total script execution time: {format_time(end_script_time - start_script_time)}s.")
    except Exception as e:
        # Global exception catch to log errors
        log.error(e)
        raise e

if __name__ == "__main__":
    main()
    
