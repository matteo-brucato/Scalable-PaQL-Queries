from src.config import read_config

config = read_config()

data_dbms_settings = {
    "dbname": config.get("Data DB", "dbname"),
    "username": config.get("Data DB", "username"),
    "password": config.get("Data DB", "password"),
    "host": config.get("Data DB", "host"),
    "port": config.get("Data DB", "port"),
}

exp_dbms_settings = {
    "dbname": config.get("Experiment DB", "dbname"),
    "username": config.get("Experiment DB", "username"),
    "password": config.get("Experiment DB", "password"),
    "host": config.get("Experiment DB", "host"),
    "port": config.get("Experiment DB", "port"),
}
