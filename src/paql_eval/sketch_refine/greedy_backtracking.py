# coding=utf-8
from collections import deque

import time
from copy import copy

from src.data_model.tuple import Tuple, Repr
from src.paql_eval.exceptions import TimeLimitElapsed
from src.paql_eval.sketch_refine.partial_package import PartialPackage
from src.paql_eval.sketch_refine.cplex_interface import CplexInterface
from src.paql_eval.sketch_refine.sketch_refine import SketchRefineRunInfo
from src.utils.log import debug, log



class GreedyBacktracking(object):
	delimiter = -1


	def __init__(self, search):
		self.search = search


	def run(self, start_empty):
		# Start with empty partial solution
		partial_sol = PartialPackage(self.search)

		# Start run info
		# gbt_runinfo = GreedyBacktrackRunInfo()
		# gbt_runinfo.run_start()
		self.search.current_run_info.strategy_run_info.run_start()

		# Solve with full-backtracking
		final_partial_package = self(partial_sol, start_empty, infeasible_cids=deque())

		# End run info
		# gbt_runinfo.bt_end()
		self.search.current_run_info.strategy_run_info.run_end()

		return final_partial_package


	def __call__(self, partial_p, start_empty, infeasible_cids):
		self.search.current_run_info.strategy_run_info.n_recursive_calls += 1

		# If start_empty is false, try to solve the representative tuples alone first (indicated by None)
		initial_cid = [] if start_empty else [None]

		# Always include clusters whose produced solution is infeasible (some constraints were removed)
		# or simply all clusters that are not completed yet (never solved or solved only in reduced space).
		missing_cids = [
			cid for cid in self.search.cids if
			cid not in partial_p.completed_cids and
			cid not in partial_p.infeasible_cids
			]
		missing_cids.sort(key=lambda cid: -self.search.n_tuples_per_cid[cid])

		debug(">> PARTIAL SOL: {}".format(partial_p))
		debug("REMAINING CIDS: {}".format(missing_cids))

		# [Backtracking Base Case] No need to augment anything, this is already a solution
		if not initial_cid and not missing_cids:
			return partial_p

		failed_cids = set()

		# Make it a queue
		remaining_cids = deque()
		remaining_cids.extend(initial_cid)
		remaining_cids.extend(missing_cids)
		remaining_cids.append(GreedyBacktracking.delimiter)

		while len(remaining_cids) > 0:
			self.search.check_timelimit()

			cid = remaining_cids.popleft()

			if cid==GreedyBacktracking.delimiter: continue

			if cid in partial_p.completed_cids:
				continue

			# If you already tried to solve this cluster before and it failed, so exit the paql_eval (fail)
			if cid in failed_cids:
				break

			print ">>> CID: {}".format(cid)
			debug(">> Trying augmenting cluster {} based on {}".format(cid, partial_p.completed_cids))

			# ================================================
			# AUGMENTING PARTIAL SOLUTION WITH CLUSTER "cid"
			# ================================================
			# Try to augment partial solution by solving for cluster cid
			# augmented_partial_p = self.augment(partial_sol, cid, runinfo)
			try:
				rtime = -time.time()
				augmented_partial_p = self.augment(partial_p, cid)
				rtime += time.time()
				print ">> AUGMENTING TIME: {}".format(rtime)

			except TimeLimitElapsed as e:
				self.search.current_run_info.strategy_run_info.run_end()
				raise e

			while augmented_partial_p=="REDO":
				augmented_partial_p = self.augment(partial_p, cid)

			assert isinstance(augmented_partial_p, PartialPackage)

			# ================================================
			# CLUSTER "cid" FAILURE
			# ================================================
			if cid in augmented_partial_p.infeasible_cids:
				self.search.current_run_info.strategy_run_info.n_infeasible_augmenting_problems += 1

				print ">>>> COULD NOT SOLVE CLUSTER {}".format(cid)

				# Add this cluster id to the infeasible clusters
				if cid not in infeasible_cids:
					infeasible_cids.append(cid)

				# If this is *not* the root, backtrack to parent node in paql_eval tree to prioritize infeasible
				# cluster
				# cid. If this *is* the root, then go to the next cluster to solve. At the end you may still fail if
				#  all
				# clusters fail.
				if len(partial_p.completed_cids) > 0:
					self.search.current_run_info.strategy_run_info.n_backtracks += 1
					return None

			# ================================================
			# CLUSTER "cid" SUCCESS
			# ================================================
			else:
				assert not augmented_partial_p.is_infeasible, augmented_partial_p
				assert cid is None or cid in augmented_partial_p.completed_cids,\
					"{}\n{}".format(cid, augmented_partial_p)

				if len(self.search.current_run_info.strategy_run_info.augmenting_problems_info)==0:
					# This is the initial package (first feasible partial package)
					self.search.current_run_info.strategy_run_info.initial_partial_package_generated()

				self.search.current_run_info.strategy_run_info.augmenting_problems_info.append({
					"cid": cid,
				})

				# Cid was successful, solve recursively on the remaining clusters
				debug(">> OK - CAN AUGMENT")
				print ">>>> OK, SOLVED CID {}".format(cid)

				# RECURSION
				new_partial_sol = self(augmented_partial_p, True, infeasible_cids)

				# If entire subtree is solved successfully, return solution to parent node
				if new_partial_sol is not None:
					return new_partial_sol

				# Otherwise, the subtree of the currently selected cluster "cid" failed.
				# You need to try with the next remaining clusters, but before you need to prioritize the failing
				# clusters
				else:
					assert cid not in remaining_cids, (cid, remaining_cids)

					# Prioritize this cid
					print ">>>> PRIORITIZING CIDS: {}".format(infeasible_cids)
					for inf_cid in infeasible_cids:
						if inf_cid in remaining_cids:
							remaining_cids.remove(inf_cid)
							# FIXME: TRYING THIS EDIT! BEFORE, THE FOLLOWING LINE WAS INDENTED 1 STEP LEFT
							remaining_cids.appendleft(inf_cid)

					print ">>>> COULD AUGMENT {} BUT DIDN'T SUCCEED".format(cid)

					debug(">> NO! - COULD AUGMENT BUT DIDN'T SUCCEED [{} based on {}]".format(
						cid, augmented_partial_p.completed_cids))

			failed_cids.add(cid)

		self.search.current_run_info.strategy_run_info.n_backtracks += 1

		return None


	def augment(self, partial_p, cid):
		"""
        Takes a partial solution consisting of:
        1) Clusters that have been solved in the original space (solved_original_cids)
        2) Clusters that have been solved in the reduced space only (solved_reduced_cids)
        3) Clusters that have never been solved in either space (empty_cids).

        Cluster "cid" will be now solved in original space if it was solved in reduced space.
        Cluster cid will be solved in reduced space (along with other clusters) if it was never solved before.
        Cluster cid should not have been solved in original space already, otherwise it is an error.

        Each remaining cluster (except cid) will be solved in reduced space if it was not solved in there already,
        otherwise it will be left untouched and its current solution will be used as a basis solution for the new
        problem constraints. Notice that a basis solution for a specific cluster can be either in the reduced space
        or in the original space, depending on whether that cluster was solved only in the reduced space or in the
        original space.
        """
		debug("Augment cid: {}".format(cid))

		if cid is not None and self.search.n_tuples_per_cid[cid]==1:
			cid_orig_space_tuples = self.get_original_space_tuples_from_cluster(cid)
			return self.augment_partial_solution_with_solution(
				True, partial_p, cid, [1], cid_orig_space_tuples, empty_cids=[],
				empty_cids_representatives=[])

		# Clusters that have been solved in the reduced space
		solved_reduced_cids = partial_p.get_solved_reduced_cids()

		# Clusters that have been solved in the original space
		solved_original_cids = partial_p.get_solved_original_cids()

		# TODO: CHECK THIS NEW ADDITION
		if cid in solved_original_cids:
			raise Exception("Disabled for now")
			# If cid was already solved before, then it must be an infeasible cluster (query was relaxed)
			assert cid in partial_p.infeasible_cids, (cid, partial_p.infeasible_cids)
			# In this case, we are trying to re-solve it from scratch
			partial_p.clear_reduced_space_solution_for_cluster(cid)
			partial_p.clear_original_space_solution_for_cluster(cid)
			solved_original_cids.remove(cid)
			solved_reduced_cids.discard(cid)

		# The order is: you first solve in reduced space and then in original space

		debug("R: {}".format(solved_reduced_cids))
		debug("O: {}".format(solved_original_cids))
		assert cid not in solved_original_cids

		cid_orig_space_tuples = self.get_original_space_tuples_from_cluster(cid)

		################################################################################################################
		# BASIS SOLUTION OF SOLVING CLUSTER cid
		################################################################################################################
		# This is a solution only in the reduced space. It will be used to generate cardinality bounds only.
		cid_basis_sol = []
		if cid not in partial_p.infeasible_cids:
			for r, s in partial_p.get_cluster_sol(cid):
				assert isinstance(r, Repr)
				assert not hasattr(r, "id") or r.id is None,\
					"Cluster {} was already solved in original space".format(cid)
				if s > 0:
					cid_basis_sol.append((cid, r, s))
		debug("cid basis sol: {}".format(cid_basis_sol))

		################################################################################################################
		# EMPTY CLUSTERS: Clusters never solved in any space
		################################################################################################################
		# Every cluster with empty solution will be solved in the reduced space now,
		# except cid which will be solved directly in the original space
		all_cids = self.search.n_tuples_per_cid
		empty_cids = set(
			c for c in all_cids
			if c!=cid
			and c not in solved_reduced_cids
			and c not in solved_original_cids)
		empty_cids_representatives = self.get_reduced_space_representatives_from_clusters(empty_cids)
		debug("empty: {}".format(empty_cids))

		if len(cid_orig_space_tuples) + len(empty_cids_representatives)==0:
			print "empty cids:", empty_cids
			print partial_p
			print cid
			print cid_orig_space_tuples
			print empty_cids_representatives

			raise Exception("Should not happen.")

		################################################################################################################
		# BASIS SOLUTION
		################################################################################################################
		# Every remaining cluster that has been solved in either the reduced or original space will be used as basis
		# solution, i.e., their aggregates will be used as constants to modify each constraint bound
		basis_cids = [c for c in all_cids if c!=cid and c not in empty_cids]

		# Pre-compute all aggregates of the basis solution (sums among all cids except current solving cid)
		count_basis = 0
		sums_basis = { attr: 0 for attr in self.search.query_attrs }
		for c in basis_cids:
			count_basis += partial_p.get_count(c)
			for attr in self.search.query_attrs:
				sums_basis[attr] += partial_p.get_sum(c, attr)

		# Augment using CPLEX
		cplex_interface = CplexInterface(self.search, self.search.store_lp_problems_dir)
		cid_feasible, cid_results = cplex_interface._cplex_augment(
			cid, cid_orig_space_tuples, cid_basis_sol, empty_cids_representatives, count_basis, sums_basis)

		if cid_feasible=="REDO":
			return "REDO"

		# If CPLEX problem was feasible, great! Return the new augmented partial solution
		if cid_feasible:
			augmented_partial_p = self.augment_partial_solution_with_solution(
				cid_feasible, partial_p, cid, cid_results, cid_orig_space_tuples,
				empty_cids, empty_cids_representatives)

			del cid_orig_space_tuples[:]
			del empty_cids_representatives[:]

			return augmented_partial_p

		else:
			if cid_results is None:
				# CPLEX problem was infeasible and has not been relaxed.
				augmented_partial_p = copy(partial_p)
				augmented_partial_p.set_infeasible(cid, infeasible_constraints=None)
				return augmented_partial_p

			else:
				# CPLEX problem was infeasible but it has been relaxed and solved.
				# The solution obtained is infeasible for the original problem.
				augmented_partial_p = self.augment_partial_solution_with_solution(
					cid_feasible, partial_p, cid, cid_results, cid_orig_space_tuples, empty_cids,
					empty_cids_representatives)

				del cid_orig_space_tuples[:]
				del empty_cids_representatives[:]

				return augmented_partial_p


	def augment_partial_solution_with_solution(
			self, cid_feasible, partial_p, cid, cid_solution, cid_orig_space_tuples,
			empty_cids, empty_cids_representatives):
		"""
        Takes a partial solution object and a solution (assignment) to a problem involving original tuples (and
        eventually also representative tuples), and it augment the partial solution with that.
        """
		print "AUGMENTING PARTIAL PACKAGE WITH SOLVER SOLUTION"
		assert cid_feasible
		assert cid is not None or len(cid_orig_space_tuples)==0

		augmented_partial_p = copy(partial_p)
		assert isinstance(augmented_partial_p, PartialPackage)

		# Add original space solutions for actual tuples
		if cid_feasible:
			print "AUGMENTING PARTIAL PACKAGE"
			k = 0
			for t in cid_orig_space_tuples:
				assert isinstance(t, Tuple)
				augmented_partial_p.add_original_space_solution_for_cluster(cid, t, cid_solution[k])
				k += 1

			# Fix solutions to other clusters only if this cluster was feasible
			print "FIXING SOLUTIONS TO OTHER CLUSTERS"
			# Group reduced space solutions by cid
			reduced_space_sols = { cid2: [] for cid2 in empty_cids }
			for r in empty_cids_representatives:
				assert isinstance(r, Repr)
				reduced_space_sols[r.cid] += [(r, cid_solution[k])]
				k += 1

			assert k==len(cid_solution)

			# Add reduced space solutions for representative tuples
			print "ADDING REDUCED SOLUTION FOR REPRESENTATIVES"
			for cid2, sol in reduced_space_sols.iteritems():
				for r, v in sol:
					assert isinstance(r, Repr)
					augmented_partial_p.add_reduced_space_solution_for_cluster(cid2, r, v)

				# [ PRUNING ] SPOT PREMATURE SOLUTION FOR A CLUSTER
				# If all reduced sols for a specific cluster are zero, then the cluster is completed
				# (no further actions are needed because the problem would just set the variables to zero.. SURE?)
				if cid_feasible and sum(v for r, v in sol)==0:
					assert isinstance(r, Repr)
					debug("Premature solution in original space for cluster {}".format(cid2))
					# FIXME: Is the following correct?
					for i in xrange(self.search.n_tuples_per_cid[cid2]):
						# FIXME: I WAS WORKING TO FIX THIS.
						augmented_partial_p.add_original_space_solution_for_cluster(cid2, None, 0)
						debug("Added original-space solution for cluster {}, tuple set to 0".format(cid2))

		else:
			raise NotImplementedError("Relaxing query not implemented yet.")

		print "RETURNING AUGMENTED PACKAGE"

		return augmented_partial_p


	def get_original_space_tuples_from_cluster(self, cid):
		if cid is None:
			return []

		# NOTE: A note about this ORDER BY id. The way this class manages the data would not need this ORDER BY id
		# here, but I'm adding it because ilp_solver_interface uses it. The reason why it uses it is because that
		# class uses the Package class. So I must use ORDER BY id here also in order to get comparable running times.
		# Otherwise I could simply remove it from here. Here's why Package instead needs ORDER BY id. The current
		# implementation of a Package relies on the fact that the tuples are read from the DB with an ORDER BY id
		# clause. Then incremental "seq" numbers are assigned and used to refer to the tuples with sequential numbers,
		#  instead of using the "id" from the tuples. This had the benefit of simplifying the Package implementation.
		# However, it would be better one day to change this as follows: the Package class would not store anything,
		# but only instruct the DB to store a candidate package in the DB. This way it won't matter whether you ORDER
		# BY id or not, because you would never need to keep data in the class, not even then sequential ids.
		sql = (
			"SELECT D.id, {attrs} "
			"FROM {S}.{D} D "
			"WHERE D.cid = {cid} "
			"ORDER BY D.id").format(
			S=self.search.schema_name,
			D=self.search.table_name,
			attrs=", ".join("D.{}".format(attr) for attr in self.search.query_attrs),
			cid=cid)

		# NOTE: This loads all tuples from cluster cid into main memory
		res = self.search.db.sql_query(sql, cid)

		orig_space_tuples_for_cid = []
		for t in res:
			orig_space_tuples_for_cid.append(Tuple(attrs=["id"] + self.search.query_attrs, record=t))

		# Assertion
		if len(orig_space_tuples_for_cid)!=self.search.n_tuples_per_cid[cid]:
			print sql
			print "Tuples from cluster {}: {}".format(cid, len(orig_space_tuples_for_cid))
			raise AssertionError

		return orig_space_tuples_for_cid


	def get_reduced_space_representatives_from_clusters(self, cids):
		if len(cids)==0:
			return []

		partitioning_attrs = ["cid"] + self.search.attrs

		res = []
		log("Loading representatives...")
		for r in self.search.db.sql_query(
				"SELECT * FROM {SR}.{R}".format(
					SR=self.search.sr_schema,
					R=self.search.repr_table_name)):

			if r.cid in cids:
				res.append(Repr(attrs=partitioning_attrs, record=r))

		return res



class GreedyBacktrackRunInfo(object):
	def __init__(self, *args, **kwargs):
		self.n_recursive_calls = 0
		self.n_infeasible_augmenting_problems = 0
		self.n_backtracks = 0
		self._total_wallclock_time = 0
		self._total_cputicks_time = 0
		self._initial_pack_wc_time = 0
		self._initial_pack_ct_time = 0
		self.cplex_run_info = []

		# Used by the approach that copes with false infeasible problems
		self.reclustering_info = []

		self.augmenting_problems_info = []


	def __str__(self):
		res = [
			"Full Backtracking Algorithm — Run Info:",
			"N recursive calls: {}".format(self.n_recursive_calls),
			"N solved problems: {}".format(len(self.cplex_run_info)),
			"N infeasible augmenting problems: {}".format(self.n_infeasible_augmenting_problems),
			"N backtracks: {}".format(self.n_backtracks),
			"Max problem size: {} vars".format(max(stat.cplex_problem_size for stat in self.cplex_run_info)),
			"Initial package wallclock time: {} s".format(self.initial_package_wallclock_time),
			"Initial package cputicks time: {} s".format(self.initial_package_cputicks_time),
			"Total wallclock time: {} s".format(self.total_wallclock_time),
			"Total cputicks time: {} s".format(self.total_cputicks_time),
			"Total wallclock time spent re-partitioning: {} s".format(
				sum(infos["wallclock_time"] for infos in self.reclustering_info)),
		]
		return "\n".join(res)


	def run_start(self):
		self._total_wallclock_time = -time.time()
		self._total_cputicks_time = -time.clock()


	def run_end(self):
		self._total_wallclock_time += time.time()
		self._total_cputicks_time += time.clock()


	def initial_partial_package_generated(self):
		self._initial_pack_wc_time = self._total_wallclock_time + time.time()
		self._initial_pack_ct_time = self._total_cputicks_time + time.time()


	@property
	def total_wallclock_time(self):
		return self._total_wallclock_time - sum(st.wallclock_time_to_store for st in self.cplex_run_info)


	@property
	def total_cputicks_time(self):
		return self._total_cputicks_time - sum(st.cputicks_time_to_store for st in self.cplex_run_info)


	@property
	def initial_package_wallclock_time(self):
		return self._initial_pack_wc_time - sum(st.wallclock_time_to_store for st in self.cplex_run_info)


	@property
	def initial_package_cputicks_time(self):
		return self._initial_pack_ct_time - sum(st.cputicks_time_to_store for st in self.cplex_run_info)
