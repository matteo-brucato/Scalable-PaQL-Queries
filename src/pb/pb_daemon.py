#!/usr/bin/env python
import atexit
import os
import threading
import time

import subprocess

import sys

import signal
import uuid
from collections import OrderedDict

import psutil

pb_home = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(pb_home)

from src.dbms.utils import db_quick_connect
from src.experiments.settings import result_data_directory_path


manager_uuid = uuid.uuid4().hex


def check_current_running_experiments(expdb):
	sql = "SELECT *, NOW() - starttime AS time_elapsed " \
		  "FROM exp " \
		  "WHERE status = 'running' " \
		  "ORDER BY starttime, id "
	exps = []

	for exp in list(expdb.sql_query(sql)):
		if exp.pid is not None:
			# Just check if process is still alive
			try:
				# print "Checking pid = {}".format(exp.pid)
				# os.kill(exp.pid, 0)
				proc = psutil.Process(exp.pid)
			except psutil.NoSuchProcess:
				# print "pid is dead"
				if exp.manager_uuid != manager_uuid:
					set_experiment_as_error(expdb, exp, "daemon crashed")
				else:
					raise Exception
			else:
				if proc.status() == psutil.STATUS_ZOMBIE:
					set_experiment_as_error(expdb, exp, "zombie")
				elif proc.status() == psutil.STATUS_RUNNING:
					# print "pid is alive and running"
					exps.append(exp)
				else:
					raise Exception("Process (pid={}) status `{}' not supported.".format(exp.pid, proc.status()))

		else:
			set_experiment_as_error(expdb, exp, "no pid")

	return exps


def get_next_scheduled_experiments(expdb):
	exps = list(expdb.sql_query("SELECT * FROM exp WHERE status = 'scheduled' ORDER BY addtime, id"))
	# if len(exps) >= 1:
	return exps
	# else:
	# 	return None


def set_experiment_as_running(expdb, exp, pid, stdout, stderr):
	expdb.sql_update("UPDATE exp SET "
	                 "status = 'running', starttime = NOW(),"
	                 "pid=%s, stdout=%s, stderr=%s,"
	                 "manager_uuid=%s "
	                 "WHERE id = %s",
	                 pid,
	                 os.path.relpath(stdout.name),
	                 os.path.relpath(stderr.name),
	                 manager_uuid,
	                 exp.id)
	expdb.commit()


def _set_experiment_endtime(expdb, exp):
	expdb.sql_update("UPDATE exp SET endtime = NOW() WHERE id = %s", exp.id)
	expdb.commit()


def set_experiment_as_success(expdb, exp):
	_set_experiment_endtime(expdb, exp)
	expdb.sql_update("UPDATE exp SET status = 'success' WHERE id = %s", exp.id)
	expdb.commit()


def set_experiment_as_error(expdb, exp, error):
	_set_experiment_endtime(expdb, exp)
	expdb.sql_update("UPDATE exp SET status = 'error', error = %s "
	                 "WHERE id = %s", error, exp.id)
	expdb.commit()


def cleanup(running_processes):
	for p in running_processes.itervalues():
		try:
			p["proc"].kill()
		except OSError:
			pass



def main():
	expdb = db_quick_connect("exp")

	running_processes = OrderedDict()

	atexit.register(cleanup, running_processes)

	while True:
		expdb.commit()

		# Currently running experiments
		current_exps = check_current_running_experiments(expdb)

		# Next-up experiment
		next_exps = get_next_scheduled_experiments(expdb)

		# If some processes died, remove them running processes
		current_exps_set = set(e.id for e in current_exps)
		for pid in running_processes.iterkeys():
			if running_processes[pid]["id"] not in current_exps_set:
				del running_processes[pid]

		# Asserts
		assert current_exps_set == set(p["id"] for p in running_processes.itervalues()), \
			(current_exps_set, set(p["id"] for p in running_processes.itervalues()))

		# Prints
		if len(current_exps) or len(next_exps):
			print
		if len(current_exps):
			print "There are {} currently running experiments{}".format(
				len(current_exps), ":" if len(current_exps) else ".")
			for i, e in enumerate(current_exps, start=1):
				print "R{}) Experiment(id={}) has been running for {}".format(i, e.id, e.time_elapsed)
		if len(next_exps):
			print "There are {} scheduled experiments{}".format(
				len(next_exps), ":" if len(next_exps) else ".")
			for i, e in enumerate(next_exps, start=1):
				print "S{}) Experiment(id={}) {} {}".format(i, e.id, e.exp_dir, e.exp_file)

		if len(current_exps) == 0 and len(next_exps) >= 1:
			exp = next_exps[0]
			print "Preparing to run Experiment(id={}) {} {}...".format(
				exp.id, exp.exp_dir, exp.exp_file)

			args = [ "/usr/bin/env", "python", "-m", "src.experiments.run_experiments", "--run-now",
			         exp.exp_dir, exp.exp_file ] + exp.args.split(" ") + [
				("--expdb", exp.expdb) if exp.expdb is not None else "",
				"--poolsize", str(exp.poolsize),
				# "--repetitions", str(exp.repetitions),
				"--logging_level", str(exp.logging_level),
				"--verbose" if exp.set_verbose else "" ]

			uid = uuid.uuid4().hex
			stdout_file_id = "{}_{}".format(exp.id, uid)
			stderr_file_id = "{}_{}".format(exp.id, uid)
			stdout = open(os.path.join(result_data_directory_path, "stdout", stdout_file_id), "wb", 1)
			stderr = open(os.path.join(result_data_directory_path, "stderr", stderr_file_id), "wb", 1)

			print "Running Experiment(id={}):".format(exp.id)
			print " ".join(args)

			p = subprocess.Popen(args, env=os.environ, stdout=stdout, stderr=stderr)
			set_experiment_as_running(expdb, exp, p.pid, stdout, stderr)

			running_processes[p.pid] = {
				"id": exp.id,
				"proc": p,
				"stdout": stdout,
				"stderr": stderr,
			}

			def callback(p):
				while p.poll() is None:
					try:
						p.wait()

					except (KeyboardInterrupt, SystemExit) as e:
						print (KeyboardInterrupt, SystemExit)
						p.send_signal(signal.SIGINT)
						# sys.exit(-1)
						set_experiment_as_error(expdb, exp, str(e))

					except Exception as e:
						print Exception
						p.send_signal(signal.SIGTERM)
						# raise e
						set_experiment_as_error(expdb, exp, str(e))

					else:
						if p.returncode == 0:
							set_experiment_as_success(expdb, exp)
						else:
							set_experiment_as_error(expdb, exp, "returncode = {}".format(p.returncode))

					# finally:
					# 	set_experiment_endtime(expdb, exp)

			# Set callback function
			thread = threading.Thread(target=callback, args=(p,))
			thread.start()

			# else:
			# 	print "No new experiments to run."

		time.sleep(5)

	expdb.close()


if __name__ == "__main__":
	main()
