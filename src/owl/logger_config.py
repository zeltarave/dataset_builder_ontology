import logging
import os

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Configura un logger che scrive sia su file che sulla console."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    

    if not logger.handlers:
        # Includiamo data, livello e messaggio
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Handler per la scrittura su file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

        # Handler per la console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger
