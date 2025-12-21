import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
