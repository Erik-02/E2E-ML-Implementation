"""
This file is used to load our environment variables. 
"""

import os
from dotenv import load_dotenv

# Load environment variables
def load_variables():
    # Define global variables for easy use
    global EIA_KEY, WANDB_KEY, DB_HOSTNAME, DB_DATABASE, DB_USERNAME, DB_PASSWORD
    load_dotenv() # This reads the environment variables inside .env
    EIA_KEY = os.getenv('EIA_KEY')
    WANDB_KEY = os.getenv('WANDB_KEY')
    DB_HOSTNAME = os.getenv('DB_HOSTNAME')
    DB_DATABASE = os.getenv('DB_DATABASE')
    DB_USERNAME = os.getenv('DB_USERNAME')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
