"""
This file contains all of the utilities needed to run some of the scripts, including:
- Logger: To log all of the processes for if there were to arise an error
"""

# Necessary libraries
import logging

# Create a logger to use
def get_logger(name) -> logging.Logger:
    """
    Template for getting a logger.

    Args:
        name: Name of the logger.

    Returns: Logger.
    """

    logging.basicConfig(level=logging.INFO, filename='Process.log', format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    logger = logging.getLogger(name)

    return logger