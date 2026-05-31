import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])

# Custom StreamHandler
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
