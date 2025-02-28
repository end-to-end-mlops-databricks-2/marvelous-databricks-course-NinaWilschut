import logging
import os

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

print(os.getcwd())

# config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# logger.info("Configuration loaded:")
# logger.info(yaml.dump(config, default_flow_style=False))


# fetch dataset
# Works both locally and in a Databricks environment
filepath = "../data/data.csv"
# Load the data
df = pd.read_csv(filepath)
print(df)
