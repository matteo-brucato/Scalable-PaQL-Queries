import logging


_verbose = False


# formatter = logging.Formatter(
# logging_format = \
# 	"%(asctime)s - %(created)f - %(filename)s - %(funcName)s - %(name)s - %(levelname)s - %(message)s"
# logging.basicConfig(format=logging_format)


def set_logging_level(logging_level, verbose=False):
	global _verbose
	# logging.basicConfig(format="%(levelname)s: %(message)s", level=logging_level)
	logging.basicConfig(format="%(message)s", level=logging_level)
	_verbose = verbose


def log(*args):
	"""
	Use this to print strictly necessary info about algorithm progress.
	"""
	if logging.getLogger().isEnabledFor(logging.INFO):
		logging.info(" ".join(str(arg) for arg in args))
		logging.getLogger().handlers[0].flush()


def verbose_log(*args):
	"""
	Use this to print "more info" than the strictly necessary info (but not at the level of debug messages)
	"""
	if _verbose and logging.getLogger().isEnabledFor(logging.INFO):
		logging.info(" ".join(str(arg) for arg in args))
		logging.getLogger().handlers[0].flush()


def debug(*args):
	if logging.getLogger().isEnabledFor(logging.DEBUG):
		logging.debug(" ".join(str(arg) for arg in args))
		logging.getLogger().handlers[0].flush()


def warn(*args):
	if logging.getLogger().isEnabledFor(logging.WARNING):
		logging.warning(" ".join(str(arg) for arg in args))
		logging.getLogger().handlers[0].flush()
