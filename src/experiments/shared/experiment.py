from abc import ABCMeta, abstractmethod
import glob
import os
import datetime
import shutil
import sys
from argparse import ArgumentParser

from src.experiments.settings import result_data_directory_path




def create_dir(dirpath):
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)


def get_next_id(test_result_pathname):
	ids = []
	for f in glob.glob(test_result_pathname + "*"):
		ids.append(int(os.path.basename(f).split("_")[-1]))
	return max(ids) + 1 if ids else 0



class Experiment(object):
	# This class is abstract, meaning that instances of this class cannot be created -- only of its subclasses.
	__metaclass__ = ABCMeta


	def __init__(self, pool, src_folder_name, src_file_name):
		self._info_file = None
		self._rawresult_file = None
		self._test_results_dir = None
		self._time_started = None

		self.pool = pool
		self.src_folder_name = src_folder_name
		self.src_file_name = src_file_name
		self.experiment_category = src_folder_name
		self.experiment_name = src_file_name
		# TODO: You may change this to an incremental number within the same experiment folder (take max existing +1)
		self.id_string = str(datetime.datetime.now()).replace(" ", "_")

		# The dbms where input data resides, will be set during run
		self.datadb = None

		# self.args, self.other_args = None, None
		self.args = None
		self.other_args = None

		self.arg_parser = ArgumentParser(description="Pachage Query Test Runner.")


	def set_datadb(self, datadb):
		self.datadb = datadb


	def test_results_dirpath(self):
		return self._test_results_dir


	def init_data_folder(self, test_name="test"):
		test_result_pathname = "{}/{}/{}/{}".format(
			result_data_directory_path, self.experiment_category, self.experiment_name, test_name)
		next_id = get_next_id(test_result_pathname)
		self._test_results_dir = "{}_{}".format(test_result_pathname, next_id)
		create_dir(self._test_results_dir)

		# INFO file
		self.open_info_file()
		self.print_info_file("Test name: {}".format(self.experiment_name))
		self.print_info_file("Command line arguments: {}".format(" ".join(sys.argv)))
		self.print_info_file("Python class name: {}".format(self.__class__.__name__))
		self.print_info_file("Test description: {}".format(self.description))
		self.print_info_file("Number of runs: {}".format(self.repetitions))
		self.print_info_file("Started on: {}".format(self._time_started))

		# Raw result data file (save results while still running)
		self.open_rawresult_file()


	@property
	@abstractmethod
	def description(self):
		raise NotImplemented("This class should be never used directly.")


	def open_info_file(self):
		self._info_file = open("{}/INFO.txt".format(self._test_results_dir), "w")


	def close_info_file(self):
		self._info_file.close()


	def print_info_file(self, strinfo):
		print >> self._info_file, strinfo
		self._info_file.flush()


	def open_rawresult_file(self):
		self._rawresult_file = open("{}/RAW_RESULTS.dat".format(self._test_results_dir), "w")
		self.print_rawresult_file(
			"# This file contains experimental results printed whenever available during computation.\n"
			"# It might contain partial results and it might be incomplete, but it's useful for retrieving\n"
			"# results from aborted computations.\n"
			"###############################################################################################"
		)


	def close_rawresult_file(self):
		self._rawresult_file.close()


	def print_rawresult_file(self, strinfo):
		print >> self._rawresult_file, strinfo
		self._rawresult_file.flush()


	def start(self):
		self._time_started = datetime.datetime.now()
		self.set_args()


	def __call__(self, args):
		self.parse_args(args)
		# Repeat experiment various times
		for rep in xrange(self.args.repetitions):
			self.run()


	def finish(self):
		# Finish
		if self._info_file and not self._info_file.closed:
			time_finished = datetime.datetime.now()
			elapsed_time = time_finished - self._time_started
			print "Experiment successfully finished on: {}".format(time_finished)
			print "Total time elapsed: {} seconds.".format(elapsed_time)
			self.print_info_file("Finished successfully on: {}".format(time_finished))
			self.print_info_file("Total time elapsed: {} seconds.".format(elapsed_time))
			self.close_info_file()
		if self._rawresult_file and not self._rawresult_file.closed:
			self.print_rawresult_file(
				'###############################################################################################\n'
				'# The data is complete. No abortions occurred.'
			)
			self.close_rawresult_file()


	def destroy(self):
		if os.path.isdir(self._test_results_dir):
			shutil.rmtree(self._test_results_dir)


	@abstractmethod
	def run(self):
		raise NotImplemented


	def parse_args(self, args):
		self.args, self.other_args = self.arg_parser.parse_known_args(args)


	@abstractmethod
	def set_args(self):
		self.arg_parser.add_argument("--repetitions", default=1, type=int)
