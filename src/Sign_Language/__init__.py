import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"

log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# logging.basicConfig(
#     filename=log_filepath,
#     level=logging.INFO,
#     format=logging_str,

#     handlers=[
#         logging.FileHandler(log_filepath),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# logger = logging.getLogger("ASLclasifierLogger")


# Create a logger
logger = logging.getLogger("ASLclasifierLogger")
logger.setLevel(logging.INFO)

# Create a file handler and set the formatter
file_handler = logging.FileHandler(log_filepath)
file_handler.setFormatter(logging.Formatter(logging_str))

# Create a stream handler and set the formatter
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter(logging_str))

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)