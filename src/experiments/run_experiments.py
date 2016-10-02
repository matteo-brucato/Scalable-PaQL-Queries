from importlib import import_module
from argparse import ArgumentParser
import logging
from multiprocessing import Pool
import traceback
import sys
from src.dbms.dbms_settings import exp_dbms_settings
from src.dbms.utils import db_quick_connect
from src.experiments.shared.experiment import Experiment

from src.utils.log import set_logging_level



def main():
	parser = ArgumentParser(description="Pachage Query Test Runner.")
	parser.add_argument('experiment_folder')
	parser.add_argument('experiment_file', nargs='?', default="main")
	parser.add_argument("--run-now", action="store_true")
	parser.add_argument("--expdb", dest="experiment_dbname", default=None)
	parser.add_argument('--poolsize', default=1, type=int)
	parser.add_argument("--logging-level", dest="logging_level", default=logging.INFO)
	parser.add_argument("--debug",
	                    help='Print lots of debugging statements',
	                    action="store_const", dest="logging_level", const=logging.DEBUG, default=logging.INFO)
	parser.add_argument("-V", "--verbose",
	                    help='Be verbose',
	                    action="store_const", dest="verbose", const=True, default=False)

	parser.set_defaults(run_now=False)

	# Parse arguments
	args, other_args = parser.parse_known_args()

	# Override experimental DB name
	if args.experiment_dbname is not None:
		exp_dbms_settings["dbname"] = args.experiment_dbname

	if not args.run_now:
		# Schedule experiment
		expdb = db_quick_connect("exp")
		expdb.sql_update(
			"INSERT INTO exp(exp_dir, exp_file, args, expdb, poolsize, logging_level, set_verbose) "
			"VALUES (%s, %s, %s, %s, %s, %s, %s)",
			args.experiment_folder,
			args.experiment_file,
			" ".join(other_args),
			args.experiment_dbname,
			args.poolsize,
			args.logging_level,
			args.verbose,
		)
		expdb.commit()
		expdb.close()

	else:
		# Set logging level
		set_logging_level(logging_level=args.logging_level, verbose=args.verbose)

		if args.poolsize > 1:
			pool = Pool(processes=args.poolsize)  # start a pool of parallel worker processes
		else:
			pool = None

		exp_module_name = "src.experiments." + args.experiment_folder + "." + args.experiment_file
		experiment_module = import_module(exp_module_name)

		for ExperimentClass in experiment_module.tests:
			experiment = ExperimentClass(pool, args.experiment_folder, args.experiment_file)
			assert isinstance(experiment, Experiment)

			print "#########################################################"
			print "Experiment Name:"
			print "\t", experiment.experiment_name
			print "Experiment Description:"
			print "\t", experiment.description

			print "---------------------------------------------------------"
			print "# Experiment Running..."
			print "---------------------------------------------------------"

			# Run the test experiment
			experiment.start()
			try:
				experiment(other_args)
			except Exception as e:
				print "Exception during experiment"
				traceback.print_exc(file=sys.stdout)
				raise e
			experiment.finish()

			print "---------------------------------------------------------"
			print "# Experiment Finished."
			print "#########################################################"

		print "End of All Experiments."




if __name__ == '__main__':
	main()
