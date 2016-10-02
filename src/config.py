import ConfigParser
import os


def read_config():
	config = ConfigParser.SafeConfigParser()
	if "PB_CONF" not in os.environ:
		raise Exception("Export env variable PB_CONF to a config name, or run `pb set <config-name>'.")

	config_name = os.environ["PB_CONF"]

	pb_home = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

	config.read(os.path.join(pb_home, "{}.cfg".format(config_name)))

	return config
