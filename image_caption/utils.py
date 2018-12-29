import logging
import sys


def setup_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
