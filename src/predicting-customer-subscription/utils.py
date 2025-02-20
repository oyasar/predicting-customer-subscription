import logging

def logger():
    """
    This function returns a logger object
    that can be used to log messages.
    Returns:

    """
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)