import os
from src.config import read_config

config = read_config()

result_data_directory_path = \
	os.path.abspath(config.get("Folders", "result_data_directory_path"))

datasets_directory_path = \
	os.path.abspath(config.get("Folders", "datasets_directory_path"))
