import os
import sys
from datetime import datetime
import logging

# Create log file with current date time
LOG_FILE = f"{datetime.now().strftime('%m-%d-%y-%H-%M-%S')}.log"
# creating logs folder in cwd under that we can store .log file
log_path = os.path.join(os.getcwd(),'logs',LOG_FILE)
# making dir
os.makedirs(log_path,exist_ok=True)
LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info('Logging Testing')