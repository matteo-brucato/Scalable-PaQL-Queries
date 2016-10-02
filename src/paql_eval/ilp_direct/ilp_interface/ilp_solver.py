# coding=utf-8
import collections
import copy
import re
from abc import abstractmethod
from math import floor, ceil

import math

from src.paql.aggregates import *
from src.paql.constraints import *
from src.paql.objectives import *
from src.paql.package_query import PackageQuery
from src.paql_eval.search import InfeasiblePackageQuery
from src.utils.log import *
from src.utils.utils import op_to_opstr



class LinearConstraint(object):
	"""
    Solver-independent representation of a linear constraint.
    """


	def __init__(self, cid, vals_func, vars_func, ugc=None, op=None, rhs=None):
		assert (ugc is not None and op is None and rhs is None) or\
		       (ugc is None and op is not None and rhs is not None)

		assert (type(cid) is int and 0 <= cid) or type(cid) is str
		if ugc is not None:
			assert isinstance(ugc, UGlobalConstraint)

		# Constraint (0-based) ID
		# NOTE: It must be unique within the same ILP problem!
		self.cid = cid

		# Corresponding (un-coalesced) global constraint, if there is one
		self.ugc = ugc

		# A function that returns an iterator through the variables of this linear constraint
		self.get_vars_function = vars_func

		# A function that returns an iterator through the coefficients of this linear constraint
		self.get_coeff_function = vals_func

		if ugc is not None:
			self.op = ugc.op
			self.rhs = ugc.rhs
		else:
			self.op = op
			self.rhs = rhs


	def __str__(self):
		if self.ugc is not None:
			return "Linear Constraint {}: {} {} {}".format(
				self.cid, self.ugc, op_to_opstr(self.ugc.op), self.ugc.rhs)
		else:
			return "Linear Constraint {}".format(self.cid)


	def __eq__(self, other):
		return self.cid==other.cid


	def __ne__(self, other):
		return self.cid!=other.cid


	def __hash__(self):
		return hash(self.cid)


	def get_name(self):
		return str(self.cid)


	def get_coefficients(self):
		return self.get_coeff_function[0](*self.get_coeff_function[1])


	def get_variables(self):
		return self.get_vars_function[0](*self.get_vars_function[1])



class ILPPackageIgnored(Exception):
	pass



class NotILPError(Exception):
	pass



class ILPPackageProblem(object):
	__metaclass__ = ABCMeta

	n_solver_solve_calls = 0
	last_contraint_id = -1  # Note: This could also be class-specific because we only need uniqueness within each ILP


	@property
	def problem(self):
		return self._linear_problem


	@problem.setter
	def problem(self, problem):
		self._linear_problem = problem


	@property
	def n_tuple_variables(self):
		return self.search.N


	def __init__(self, search, **kwargs):
		self.search = search
		self.db = search.db

		# The linear problem (specific type depends on specific solver being used)
		self._linear_problem = None

		# A collection of vectors, one for each table attribute used in the query
		self.coeff_vectors = None

		# Indexes (as 0-based IDs) of the variables that correspond to the dataset input tuples
		self.tuple_variables = None

		self.store_all_solved_problems = kwargs.get("store_all_solved_problems", False)

		self.linear_constraints = None
		self.removed_linear_constraints = None
		self.conflicting_linear_constraints = None


	@property
	@abstractmethod
	def problem_type(self):
		raise NotImplementedError


	@abstractmethod
	def add_variables(self, n_variables, lb=None, ub=None, var_type=None):
		raise NotImplementedError


	@abstractmethod
	def init_ilp(self):
		self.linear_constraints = []
		self.removed_linear_constraints = []
		self.conflicting_linear_constraints = None


	@abstractmethod
	def set_objective_sense(self, sense):
		raise NotImplemented


	@abstractmethod
	def get_objective_sense(self):
		raise NotImplemented


	@abstractmethod
	def set_linear_objective(self, variables, coefficients):
		raise NotImplemented


	@abstractmethod
	def add_linear_constraints(self, linear_constraints):
		raise NotImplemented


	@abstractmethod
	def remove_linear_constraints(self, linear_constraints):
		raise NotImplemented


	@abstractmethod
	def compute_conflicting_linear_constraints(self):
		raise NotImplemented


	@abstractmethod
	def get_optimal_package(self, packages_to_ignore):
		raise NotImplemented


	@abstractmethod
	def get_one_feasible_package(self, packages_to_ignore):
		raise NotImplemented


	@abstractmethod
	def get_tuple_solution_values(self):
		raise NotImplemented


	@abstractmethod
	def get_linear_problem_copy(self):
		raise NotImplemented


	@abstractmethod
	def get_linear_problem_string(self):
		raise NotImplemented


	@abstractmethod
	def store_linear_problem_string(self, dirpath):
		raise NotImplemented


	def new_linear_constraint(self, vals_func, vars_func, op, rhs, cid=None):
		self.last_contraint_id += 1 if cid is None else 0
		return LinearConstraint(
			cid=self.last_contraint_id if cid is None else cid,
			vals_func=vals_func,
			vars_func=vars_func,
			op=op,
			rhs=rhs)


	def add_tuple_variables(self, var_type=None):
		self.tuple_variables =\
			self.add_variables(n_variables=self.n_tuple_variables, lb=0.0, ub=1.0, var_type=var_type)


	def load_ilp_from_paql(self, problem_type="auto"):
		assert isinstance(self.search.query, PackageQuery)

		# Init solver problem
		self.init_ilp()

		# Set tuple variables
		if problem_type.lower()=="lp":
			self.add_tuple_variables("continuous")

		elif problem_type.lower()=="auto":
			self.add_tuple_variables("auto")

		else:
			raise Exception("Problem type '{}' not supported.".format(problem_type))

		if len(self.tuple_variables)==0:
			return

		# Get all query attributes
		attrs = self.search.query.get_attributes()

		if len(attrs) > 0:
			# Retrieve query data (aka the constraint coefficients) from DBMS, based on query attributes.
			# NOTE: This is the fastest way (also used in sketch_refine method), but it's not memory efficient because
			# it loads up the entire dataset table in main memory, to then feed it to CPLEX. Basically, this way you
			# will need 2N space, N to store this variable, and N to load the CPLEX problem (which also resides
			# in main memory).
			# NOTE: If you change this implementation, and make it (for instance) more memory efficient, there is a
			# high chance that the SketchRefine method will have to be revised as well to mimic whatever you did here.
			#  So, in the future, to keep things more organized, you should try to find a way to avoid creating the
			# CPLEX problem in the SketchRefine method itself, and always use a single API to create any CPLEX problem
			#  in the entire package.
			sql = "SELECT {attrs} FROM {S}.{R} ORDER BY id".format(
				attrs=",".join(attrs),
				S=self.search.schema_name,
				R=self.search.table_name)
			print sql
			data_iterator = self.db.sql_query(sql)

			self.load_data(data_iterator, attrs)

			# Set the objective type (minimization or maximization)
			self.set_paql_objective_sense()

			# Add optimization objective
			self.set_paql_objective()

			# Add base constraints
			self.add_base_constraints()

			# Add global constraints
			self.add_global_constraints()

			print "TODO: YOU SHOULD PROBABLY CLEAN DATA HERE, BUT FOR NOW I'M DISABLING IT"
			self.clear_data()

			# Problem must be a (M)ILP ((Mixed) Integer Linear Program)
			if self.problem_type!="MILP" and self.problem_type!="ILP":
				raise NotILPError("Not a (M)ILP problem. Problem type is {}".format(self.problem_type))

		else:
			warn("The package query does not refer to any attribute, so the problem has no constraints!")


	def load_data(self, data_iterator, attrs):
		"""
        Load up the necessary data (some columns of the whole input table) in main memory.

        This method iterates through the input data *once* and saves (in memory)
        a data vector for each query attribute. This view of the data is the more
        convenient for feeding the problem to the solver.
        An extra vector contains n 1's, for convenient use for cardinality constraints.
        This data strucure is the only point where we store the data in memory.
        """
		self.coeff_vectors = { attr: [None] * self.n_tuple_variables for attr in attrs }

		# NOTE: This is the only moment we iterate through the data received from the DBMS
		n = 0
		for i, d in enumerate(data_iterator):
			for attr in attrs:
				self.coeff_vectors[attr][i] = getattr(d, attr)
			n += 1

		assert self.n_tuple_variables==n,\
			"Something's wrong: data size ({}) != from # tuple variables ({})!".format(
				n, self.n_tuple_variables)

		# Convenient extra vector for COUNT constraints
		self.coeff_vectors["*"] = [1] * self.n_tuple_variables


	def clear_data(self):
		"""
        Remove data from main memory.
        """
		for attr in self.coeff_vectors.iterkeys():
			del self.coeff_vectors[attr][:]
		self.coeff_vectors.clear()
		self.coeff_vectors = None


	def set_paql_objective_sense(self):
		if self.search.query.objective is not None:
			self.set_objective_sense(self.search.query.objective.sense)
		else:
			self.set_objective_sense(ObjectiveSenseMIN)


	def set_paql_objective(self):
		assert isinstance(self.search.query, PackageQuery)
		assert self.search.query.objective is None or isinstance(self.search.query.objective, PackageQueryObjective)

		log("SETTING ILP OBJECTIVE")

		if not self.search.query.objective:
			pass

		elif self.search.query.objective.is_simple_linear():
			aggregate = self.search.query.objective.get_aggregate()
			assert isinstance(aggregate, SingleArgAggr) and aggregate.is_simple_linear()
			attr = aggregate.args[0]

			self.set_linear_objective(self.tuple_variables, self.coeff_vectors[attr])

		else:
			raise Exception("Objective not supported: {}".format(self.search.query.objective))


	def add_base_constraints(self):
		pass  # Assuming you already prepared the data with correct tuples


	def __get_void_linear_constraint(self):
		"""
        This constraint is COUNT(*) >= 0, which is always true.
        """
		return self.new_linear_constraint(
			vals_func=(lambda: self.coeff_vectors["*"], ()),
			vars_func=(lambda: self.tuple_variables, ()),
			op=operator.ge,
			rhs=0)


	def __get_simple_linear_constraint(self, ugc):
		"""
        Linearizes a constraint of one of these forms:
            (SELECT COUNT(*) FROM P) <op> <rhs>
            (SELECT SUM(<col>) FROM P) <op> <rhs>
            (SELECT AVG(<col>) FROM P) <op> <rhs>
        """
		assert isinstance(ugc, UGlobalConstraint)
		assert self.coeff_vectors is not None

		aggregates = list(ugc.iter_aggregates())
		assert len(aggregates)==1
		aggregate = aggregates[0]
		assert isinstance(aggregate, SingleArgAggr)

		if isinstance(aggregate, CountAggr) or isinstance(aggregate, SumAggr):
			coeffs = self.coeff_vectors[aggregate.arg]
			op = ugc.op
			rhs = ugc.rhs

		elif isinstance(aggregate, AvgAggr):
			coeffs = [c - ugc.rhs for c in self.coeff_vectors[aggregate.arg]]
			op = ugc.op
			rhs = 0

		else:
			raise Exception("This aggregate is not supported yet: %s" % aggregate)

		return self.new_linear_constraint(
			vals_func=(lambda x: x, (coeffs,)),
			vars_func=(lambda: self.tuple_variables, ()),
			op=op,
			rhs=rhs)


	def __generate_approx_minmax_group_by_linear_constraints(self, ugc, reduction_factor):
		"""
        Linearizes a constraint of the form:
            (SELECT COUNT(*) / (MAX(y) - MIN(y) + 1) FROM P GROUP BY x) <op> <rhs>

        The MAX(Y) and MIN(Y) are approximate (using ranges)

        The reduction factor will reduce the number of variables used to linearize MAX and MIN by that factor.
        """
		assert isinstance(ugc, UGlobalConstraint)
		assert type(reduction_factor) is int and reduction_factor >= 1

		if ugc.op is operator.ge and ugc.rhs==0:
			# (SELECT COUNT(*) / (MAX(y) - MIN(y) + 1) FROM P GROUP BY x) >= 0
			# This constraint is always true, with any package
			return

		group_by_op_tree = ugc.expr.leaf
		group_by_col = str(group_by_op_tree.group_by_expr)

		min_cols = max_cols = []
		for group_by_agg in ugc.iter_aggregates():
			if isinstance(group_by_agg, CountAggr):
				pass
			elif isinstance(group_by_agg, MinAggr):
				min_cols = group_by_agg.args
			elif isinstance(group_by_agg, MaxAggr):
				max_cols = group_by_agg.args
			else:
				raise Exception

		if len(min_cols)!=1 and min_cols!=max_cols:
			raise Exception("Constraint `{}' not supported.".format(ugc))
		min_col = min_cols[0]
		max_col = max_cols[0]

		assert min_col==max_col
		assert 0.0 <= ugc.rhs <= 1.0  # This constraint represents a ratio, so the rhs must be in [0,1]

		# At this point, the constraint is of the right form.

		mcol = min_col
		gcol = group_by_col
		r = list(self.db.sql_query(
			"SELECT {gcol}, MIN({mcol}), MAX({mcol}), COUNT(*) "
			"FROM {S}.{R} "
			"GROUP BY {gcol} "
			"ORDER BY {gcol}".format(
				mcol=mcol,
				gcol=gcol,
				S=self.search.schema_name,
				R=self.search.table_name)))
		min_max_size = [(rr[0], rr[1], rr[2], rr[3]) for rr in r]

		coeff_gcol = self.coeff_vectors[gcol]
		coeff_mcol = self.coeff_vectors[mcol]


		# Indicator function for modeling the WHERE clause
		def f1(_tix_, _x_val_):
			return 1 if _tix_==_x_val_ else 0


		where_val_dict = { }

		for j, (X_val, Y_min, Y_max, Y_size) in enumerate(min_max_size):
			Y_min = Y_min * 0  # <== FIXME: 0.0 is lb of xi's (just set ymin to 0)
			Y_max = Y_max * 1  # <== FIXME: 1.0 (or other) should come from ub of xi's (using REPEAT)
			Y_range = Y_max - Y_min + 1
			assert Y_min!=Y_max

			# Reduced Y_range
			y_range = int(ceil(float(Y_range) / reduction_factor))
			y_min = Y_min
			y_max = y_range + y_min - 1
			assert y_min!=y_max

			# Integer variables My and my that encode MAX(Y) and MIN(Y), respectively
			My = self.add_variables(1, y_min, y_max, var_type="integer")[0]
			my = self.add_variables(1, y_min, y_max, var_type="integer")[0]

			# Binary expansion of integer variables My and my
			M = self.add_variables(y_range, 0.0, 1.0, var_type="binary")
			m = self.add_variables(y_range, 0.0, 1.0, var_type="binary")

			where_val_dict[X_val] = (y_min, y_max, My, my, M, m)

			# print "Adding target constraint: {}".format(ugc.expr)
			# Σ 1(ti.x)xi - ({rhs}*r)My + ({rhs}*r)my {op} -{rhs}*r + {rhs} + {rhs}
			# Σ 1(ti.x)xi - ({rhs}*r)My + ({rhs}*r)my {op} {rhs}(2 - r)
			yield self.new_linear_constraint(
				# cid="Target_{}_j{}".format(mcol, j),
				vals_func=(
					lambda _x, _tixs:
					[f1(_tix, _x) for _tix in _tixs] +
					[-ugc.rhs * reduction_factor, ugc.rhs * reduction_factor],
					(X_val, coeff_gcol)),
				vars_func=(lambda x, y: list(self.tuple_variables) + [x, y], (My, my)),
				op=ugc.op,
				rhs=ugc.rhs * (2 - reduction_factor))

			# print "Adding binary expansion constraint: Σj Mj = 1"
			yield self.new_linear_constraint(
				# cid="BinMax_{}_j{}".format(mcol, j),
				vals_func=(lambda x: [1] * x, (y_range,)),
				vars_func=(lambda x: x, (M,)),
				op=operator.eq,
				rhs=1.0)

			# print "Adding binary expansion constraint: Σj mj = 1"
			yield self.new_linear_constraint(
				# cid="BinMin_{}_j{}".format(mcol, j),
				vals_func=(lambda x: [1] * x, (y_range,)),
				vars_func=(lambda x: x, (m,)),
				op=operator.eq,
				rhs=1.0)


		for xi, tiy, tix in izip(self.tuple_variables, coeff_mcol, coeff_gcol):
			Y_min, Y_max, My, my, M, m = where_val_dict[tix]
			Y_diff = (Y_max - Y_min) * reduction_factor

			# Index of this ti.y value in the binary expansions of My and my
			j = int(floor(float(tiy - Y_min) / reduction_factor))

			# Binary variables of expansions of My and my that corresponds to the i-th tuple ti
			mj = m[j]

			# Using these implicit constraints to create the problem constraints:
			# r*My <= MAX(Y) <= r*(My+1) - 1
			# r*my <= MIN(Y) <= r*(my+1) - 1

			######################################################
			# MAX linear formulation
			######################################################
			# MAX(Y) >= (ti.y)xi  ==>
			# (tiy)xi - r*My <= r-1
			yield self.new_linear_constraint(
				# cid="MAX_1_{}_j{}_i{}_ii{}".format(mcol, j, i, ii),
				vals_func=(lambda _tiy: [_tiy, -reduction_factor], (tiy,)),
				vars_func=(lambda _xi, _My: [_xi, _My], (xi, My)),
				op=operator.le,
				rhs=reduction_factor - 1)

			######################################################
			# MIN linear formulation
			######################################################
			# MIN(Y) >= (ti.y)xi - Y_diff*(1-mj)  ==>
			# (ti.y)xi - r*my + (Y_diff)mj <= Y_diff+r-1
			yield self.new_linear_constraint(
				# cid="MIN_2_{}_j{}_i{}_ii{}".format(mcol, j, i, ii),
				vals_func=(lambda _tiy, d: [_tiy, -reduction_factor, d], (tiy, Y_diff)),
				vars_func=(lambda _xi, _myj, _mij: [_xi, _myj, _mij], (xi, my, mj)),
				op=operator.le,
				rhs=Y_diff + reduction_factor - 1)

			# Binary variables of expansions of My and my that corresponds to the i-th tuple ti
			Mj = M[j]

			######################################################
			# MAX linear formulation
			######################################################
			# MAX(Y) <= (ti.y)xi + Y_diff*(1-Mj) + Y_diff*(1-xi)  ==>
			# (ti.y - Y_diff)*xi - r*My - Y_diff*Mj >= -2*Y_diff
			yield self.new_linear_constraint(
				# cid="MAX_2_{}_j{}_i{}_ii{}".format(mcol, j, i, ii),
				vals_func=(lambda _tiy, d: [_tiy - d, -reduction_factor, -d], (tiy, Y_diff)),
				vars_func=(lambda _xi, _My, _Mj: [_xi, _My, _Mj], (xi, My, Mj)),
				op=operator.ge,
				rhs=-(2 * Y_diff))

			######################################################
			# MIN linear formulation
			######################################################
			# MIN(Y) <= (ti.y)xi + Y_diff*(1-xi)  ==>
			# (ti.y - Y_diff)xi - r*my >= -Y_diff
			yield self.new_linear_constraint(
				# cid="MIN_1_{}_j{}_i{}_ii{}".format(mcol, j, i, ii),
				vals_func=(lambda _tiy, d: [_tiy - d, -reduction_factor], (tiy, Y_diff)),
				vars_func=(lambda _xi, _myj: [_xi, _myj], (xi, my)),
				op=operator.ge,
				rhs=-Y_diff)


	def generate_minmax_group_by_linear_constraints__rangebased(self, ugc):
		raise NotImplementedError


	def generate_box_minmax_group_by_linear_constraints(self, ugc):
		"""
        Linearizes a constraint of the form:
            (SELECT COUNT(*) FROM P GROUP BY x) ALL <op> <rhs> * (SELECT (MAX(y) - MIN(y) + 1) FROM P)
        """
		assert isinstance(ugc, UGlobalConstraint)

		if ugc.op is operator.ge and ugc.rhs==0:
			# This constraint is always true, with any package
			return

		group_by_col = str(ugc.expr.leaf.group_by_expr)

		min_cols = max_cols = []
		for group_by_agg in ugc.iter_aggregates():
			if isinstance(group_by_agg, CountAggr):
				pass
			elif isinstance(group_by_agg, MinAggr):
				min_cols = group_by_agg.args
			elif isinstance(group_by_agg, MaxAggr):
				max_cols = group_by_agg.args
			else:
				raise Exception

		if len(min_cols)!=1 and min_cols!=max_cols:
			raise Exception("Constraint `{}' not supported.".format(ugc))
		min_col = min_cols[0]
		max_col = max_cols[0]

		assert min_col==max_col
		assert min_col!=group_by_col
		assert 0.0 <= ugc.rhs <= 1.0  # This constraint represents a ratio, so the rhs must be in [0,1]

		# At this point, the constraint is of the right form.

		mcol = min_col
		gcol = group_by_col
		# print "Step 1: Read min/max({mcol}) group by {gcol} from DB".format(mcol=mcol, gcol=gcol)
		r = self.db.sql_query(
			"SELECT MIN({mcol}), MAX({mcol}) FROM {S}.{R} ".format(
				mcol=mcol,
				gcol=gcol,
				S=self.search.schema_name,
				R=self.search.table_name)).next()
		glob_min_max = (r[0], r[1])

		coeff_gcol = self.coeff_vectors[gcol]
		coeff_mcol = self.coeff_vectors[mcol]

		unique_gvals = list(self.db.sql_query("SELECT DISTINCT {gcol} FROM {S}.{R} ORDER BY {gcol}".format(
			gcol=gcol,
			S=self.search.schema_name,
			R=self.search.table_name,
		)))

		for c in self.generate_box_minmax_group_by_linear_constraints_gen(
				ugc, coeff_gcol, coeff_mcol, glob_min_max, unique_gvals):
			yield c


	def generate_box_minmax_group_by_linear_constraints_gen(self, ugc, gvals, mvals, glob_min_max, unique_gvals):
		# Indicator function for modeling the WHERE clause
		def f1(_tix_, _x_val_):
			return 1 if _tix_==_x_val_ else 0


		Y_min, Y_max = glob_min_max
		Y_min = Y_min * 0
		Y_max = Y_max * 1
		Y_range = int(math.ceil(Y_max - Y_min + 1))
		Y_diff = Y_max - Y_min
		assert Y_min!=Y_max

		My = self.add_variables(1, Y_min, Y_max, var_type="integer")[0]
		my = self.add_variables(1, Y_min, Y_max, var_type="integer")[0]
		M = self.add_variables(Y_range, 0.0, 1.0, var_type="binary")
		m = self.add_variables(Y_range, 0.0, 1.0, var_type="binary")

		# Adding binary expansion constraint: Σj Mj = 1
		yield self.new_linear_constraint(
			vals_func=(lambda x: [1] * x, (Y_range,)),
			vars_func=(lambda x: x, (M,)),
			op=operator.eq,
			rhs=1.0)

		# Adding binary expansion constraint: Σj mj = 1
		yield self.new_linear_constraint(
			vals_func=(lambda x: [1] * x, (Y_range,)),
			vars_func=(lambda x: x, (m,)),
			op=operator.eq,
			rhs=1.0)

		# Adding target constraints: For each unique xval:
		# COUNT(WHERE x=xval) =/<=/>= (MAX-MIN+1) * rhs
		# Σ 1(ti.x)xi - {rhs}My + {rhs}my {op} {rhs}
		for j, gval in enumerate(unique_gvals):
			yield self.new_linear_constraint(
				vals_func=(
					lambda _x, _tixs:
					[f1(_tix, _x) for _tix in _tixs] + [-ugc.rhs, ugc.rhs], (gval, gvals)),
				vars_func=(
					lambda _My, _my:
					list(self.tuple_variables) + [_My, _my], (My, my)),
				op=ugc.op,
				rhs=ugc.rhs)

		# Add necessary MIN/MAX constraints on each input tuple ti
		for xi, tiy, tix in izip(self.tuple_variables, mvals, gvals):
			assert xi==self.tuple_variables[xi]
			assert tiy is not None and tix is not None

			# Index of this ti.y value in the binary expansions of My and my
			j = int(math.ceil(tiy - Y_min))

			Mj = M[j]
			mj = m[j]

			######################################################
			# MAX linear formulation
			######################################################
			# My >= (ti.y)xi  ==>  (ti.y)xi - My <= 0
			yield self.new_linear_constraint(
				vals_func=(lambda _tiy: [_tiy, -1], (tiy,)),
				vars_func=(lambda _xi, _My: [_xi, _My], (xi, My)),
				op=operator.le,
				rhs=0.0)

			# My <= (ti.y)xi + Y_diff*(1-Mj) + Y_diff*(1-xi)  ==>  (ti.y - Y_diff)*xi - My - Y_diff*Mj >= -2*Y_diff
			yield self.new_linear_constraint(
				vals_func=(lambda _tiy, d: [_tiy - d, -1, -d], (tiy, Y_diff)),
				vars_func=(lambda _xi, _My, _Mj: [_xi, _My, _Mj], (xi, My, Mj)),
				op=operator.ge,
				rhs=-(2 * Y_diff))

			######################################################
			# MIN linear formulation
			######################################################
			# my >= (ti.y)xi - (Y_diff)(1-mj)  ==>  (ti.y)xi - my + (Y_diff)mj <= Y_diff
			yield self.new_linear_constraint(
				vals_func=(lambda _tiy, d: [_tiy, -1, d], (tiy, Y_diff)),
				vars_func=(lambda _xi, _myj, _mij: [_xi, _myj, _mij], (xi, my, mj)),
				op=operator.le,
				rhs=Y_diff)

			# my <= (ti.y)xi + (Y_diff)(1-xi)  ===>  (ti.y - Y_diff)xi - my >= -Y_diff
			yield self.new_linear_constraint(
				vals_func=(lambda _tiy, d: [_tiy - d, -1], (tiy, Y_diff)),
				vars_func=(lambda _xi, _myj: [_xi, _myj], (xi, my)),
				op=operator.ge,
				rhs=-Y_diff)


	def generate_strip_minmax_group_by_linear_constraints(self, ugc):
		"""
        Linearizes a constraint of the form:
            (SELECT COUNT(*) / (MAX(y) - MIN(y) + 1) FROM P GROUP BY x) <op> <rhs>
        """
		assert isinstance(ugc, UGlobalConstraint)

		if ugc.op is operator.ge and ugc.rhs==0:
			# (SELECT COUNT(*) / (MAX(y) - MIN(y) + 1) FROM P GROUP BY x) >= 0
			# This constraint is always true, with any package
			return

		group_by_col = str(ugc.expr.leaf.group_by_expr)

		min_cols = max_cols = []
		for group_by_agg in ugc.iter_aggregates():
			if isinstance(group_by_agg, CountAggr):
				pass
			elif isinstance(group_by_agg, MinAggr):
				min_cols = group_by_agg.args
			elif isinstance(group_by_agg, MaxAggr):
				max_cols = group_by_agg.args
			else:
				raise Exception

		if len(min_cols)!=1 and min_cols!=max_cols:
			raise Exception("Constraint `{}' not supported.".format(ugc))
		min_col = min_cols[0]
		max_col = max_cols[0]

		assert min_col==max_col
		assert min_col!=group_by_col
		assert 0.0 <= ugc.rhs <= 1.0  # This constraint represents a ratio, so the rhs must be in [0,1]

		# At this point, the constraint is of the right form.

		mcol = min_col
		gcol = group_by_col
		r = list(self.db.sql_query(
			"SELECT {gcol}, MIN({mcol}), MAX({mcol}), COUNT(*) "
			"FROM {S}.{R} "
			"GROUP BY {gcol} "
			"ORDER BY {gcol}".format(
				mcol=mcol,
				gcol=gcol,
				R=self.search.table_name,
				S=self.search.schema_name)))
		min_max_size = [(rr[0], rr[1], rr[2], rr[3]) for rr in r]

		coeff_gcol = self.coeff_vectors[gcol]
		coeff_mcol = self.coeff_vectors[mcol]

		for c in self.generate_strip_minmax_group_by_linear_constraints_gen(ugc, coeff_gcol, coeff_mcol, min_max_size):
			yield c


	def generate_strip_minmax_group_by_linear_constraints_gen(self, ugc, coeff_gcol, coeff_mcol, min_max_size):
		# Indicator function for modeling the WHERE clause
		def f1(_tix_, _x_val_):
			return 1 if _tix_==_x_val_ else 0


		where_val_dict = { }

		for j, (X_val, Y_min, Y_max, Y_size) in enumerate(min_max_size):
			Y_min = Y_min * 0
			Y_max = Y_max * 1
			Y_range = int(math.ceil(Y_max - Y_min + 1))

			if Y_min==Y_max:
				assert Y_size==1 and Y_min==0, (X_val, Y_min, Y_max, Y_size)
				continue

			# print "Step 2: Add new variables for MIN/MAX({}) GROUP BY {}".format(mcol, gcol)
			My = self.add_variables(1, Y_min, Y_max, var_type="integer")[0]
			my = self.add_variables(1, Y_min, Y_max, var_type="integer")[0]

			M = self.add_variables(Y_range, 0.0, 1.0, var_type="binary")
			m = self.add_variables(Y_range, 0.0, 1.0, var_type="binary")

			where_val_dict[X_val] = (Y_min, Y_max, My, my, M, m)

			# print "Adding target constraint: {}".format(ugc.expr)
			# Σ 1(ti.x)xi - {rhs}My + {rhs}my {op} {rhs}
			yield self.new_linear_constraint(
				# cid="Target_{}_j{}".format(mcol, j),
				vals_func=(
					lambda _x, _tixs:
					[f1(_tix, _x) for _tix in _tixs] + [-ugc.rhs, ugc.rhs], (X_val, coeff_gcol)),
				vars_func=(
					lambda _My, _my:
					list(self.tuple_variables) + [_My, _my], (My, my)),
				op=ugc.op,
				rhs=ugc.rhs)

			# print "Adding binary expansion constraint: Σj Mj = 1"
			yield self.new_linear_constraint(
				vals_func=(lambda x: [1] * x, (Y_range,)),
				vars_func=(lambda x: x, (M,)),
				op=operator.eq,
				rhs=1.0)

			# print "Adding binary expansion constraint: Σj mj = 1"
			yield self.new_linear_constraint(
				vals_func=(lambda x: [1] * x, (Y_range,)),
				vars_func=(lambda x: x, (m,)),
				op=operator.eq,
				rhs=1.0)

		# Add necessary MIN/MAX constraints on each input tuple ti
		for xi, tiy, tix in izip(self.tuple_variables, coeff_mcol, coeff_gcol):
			assert xi==self.tuple_variables[xi]
			assert tiy is not None and tix is not None

			if tix not in where_val_dict:
				continue

			Y_min, Y_max, My, my, M, m = where_val_dict[tix]

			j = int(math.ceil(tiy - Y_min))  # Index of this ti.y value in the binary expansions of My and my

			Y_diff = Y_max - Y_min

			Mj = M[j]
			mj = m[j]

			######################################################
			# MAX linear formulation
			######################################################
			# My >= (ti.y)xi  ==>  (ti.y)xi - My <= 0
			yield self.new_linear_constraint(
				vals_func=(lambda _tiy: [_tiy, -1], (tiy,)),
				vars_func=(lambda _xi, _My: [_xi, _My], (xi, My)),
				op=operator.le,
				rhs=0.0)

			# My <= (ti.y)xi + Y_diff*(1-Mj) + Y_diff*(1-xi)  ==>  (ti.y - Y_diff)*xi - My - Y_diff*Mj >= -2*Y_diff
			yield self.new_linear_constraint(
				vals_func=(lambda _tiy, d: [_tiy - d, -1, -d], (tiy, Y_diff)),
				vars_func=(lambda _xi, _My, _Mj: [_xi, _My, _Mj], (xi, My, Mj)),
				op=operator.ge,
				rhs=-(2 * Y_diff))

			######################################################
			# MIN linear formulation
			######################################################
			# my >= (ti.y)xi - (Y_diff)(1-mj)  ==>  (ti.y)xi - my + (Y_diff)mj <= Y_diff
			yield self.new_linear_constraint(
				vals_func=(lambda _tiy, d: [_tiy, -1, d], (tiy, Y_diff)),
				vars_func=(lambda _xi, _myj, _mij: [_xi, _myj, _mij], (xi, my, mj)),
				op=operator.le,
				rhs=Y_diff)

			# my <= (ti.y)xi + (Y_diff)(1-xi)  ===>  (ti.y - Y_diff)xi - my >= -Y_diff
			yield self.new_linear_constraint(
				vals_func=(lambda _tiy, d: [_tiy - d, -1], (tiy, Y_diff)),
				vars_func=(lambda _xi, _myj: [_xi, _myj], (xi, my)),
				op=operator.ge,
				rhs=-Y_diff)


	def __generate_global_linear_constraints(self):
		# If there's no global constraints, add a single "void" constraint COUNT(*) >= 0, which is always true
		if not self.search.query.uncoalesced_gcs:
			yield self.__get_void_linear_constraint()

		# Otherwise, add linear constraints for each of the un-coalesced PaQL global constraints
		else:
			for ugc in self.search.query.uncoalesced_gcs:
				assert isinstance(ugc, UGlobalConstraint)

				match_group_by_strip = re.match(
					"\(COUNT\(\*\)\)\/\(\(\(MAX\((?P<max_col>\w+)\)\)\-\(MIN\((?P<min_col>\w+)\)\)\)\+\(1\.0+\)\) "
					"GROUP BY (?P<groupby_col>\w+) >= (?P<rhs>.+)", str(ugc))

				match_group_by_box = re.match(
					"\(COUNT\(\*\)\)\/\(\(\(MAX\((?P<max_col>\w+)\)\)\-\(MIN\((?P<min_col>\w+)\)\)\)\+\(3\.0+\)\) "
					"GROUP BY (?P<groupby_col>\w+) >= (?P<rhs>.+)", str(ugc))

				if ugc.is_simple_linear():
					yield self.__get_simple_linear_constraint(ugc)

				elif re.match("\(COUNT\(\*\)\)/\(\(MAX\(y\)\)-\(MIN\(y\)\)\) WHERE \(x\)=\(\d+\) >= 1", str(ugc)):
					raise DeprecationWarning

				elif re.match("\(COUNT\(\*\)\)/\(\(MAX\(y\)\)-\(MIN\(y\)\)\) WHERE \(x\)=\(\d+\) <= 1", str(ugc)):
					raise DeprecationWarning

				elif match_group_by_strip is not None:
					for c in self.generate_strip_minmax_group_by_linear_constraints(ugc):
						yield c

				elif match_group_by_box is not None:
					for c in self.generate_box_minmax_group_by_linear_constraints(ugc):
						yield c

				elif str(ugc)=="(COUNT(*))/(((MAX(x))-(MIN(x)))*((MAX(y))-(MIN(y)))) <= 1":
					raise DeprecationWarning

				else:
					raise AssertionError("Unsupported global constraint: {}".format(ugc))


	def add_global_constraints(self):
		log("ADDING ILP CONSTRAINTS")
		self.add_linear_constraints(self.__generate_global_linear_constraints())


	def remove_conflicting_global_constraints(self):
		self.compute_conflicting_linear_constraints()
		self.remove_linear_constraints(self.conflicting_linear_constraints)


	def remove_all_global_constraints(self):
		self.remove_linear_constraints(self.linear_constraints)


	def gen_feasible_packages_via_powers(self):
		packages = set()

		q = collections.deque([(self.problem, None)])
		while q:
			linear_prob, linear_constraint_to_add = q.popleft()

			# Create copy of the problem
			problem = copy.copy(self)
			problem.problem = linear_prob

			if linear_constraint_to_add:
				problem.add_linear_constraint(*linear_constraint_to_add)

			# Try to get optimal solution from current problem
			try:
				optimal_package, cplex_run_info = problem.get_one_feasible_package(packages)

			except InfeasiblePackageQuery:
				debug('++++++ Infeasible')
				continue

			except ILPPackageIgnored:
				debug('++++++ Ignored')
				continue

			else:
				debug('++++++ Optimal')
				packages.add(optimal_package)
				yield optimal_package

				# Handle the case in which the base relation is totally empty
				if len(problem.tuple_variables) <= 0:
					# Do nothing, the only possible candidate was the empty package
					continue

				# The last found package is not a feasible solution anymore.
				# Add new constraints to avoid this found solution and fork
				# the problem into two sub-problems whose solutions will be unified.

				# Compute dot product of the solution values with powers of (k+2)
				# This corresponds to converting the solution from a number in base 10 to a number in base k+2
				powers = [2 ** i for i in reversed(xrange(len(problem.tuple_variables)))]
				values = problem.get_tuple_solution_values()

				cur_dot = sum(powers[i] * values[i] for i in xrange(len(powers)))

				q.append((problem.get_linear_problem_copy(), (powers, 'L', cur_dot - 1)))
				q.append((problem.get_linear_problem_copy(), (powers, 'G', cur_dot + 1)))


	def gen_feasible_packages_iteratively(self):
		"""
        This method generates all feasible pacakges as follows:
        Suppose the problem is MAXIMIZE SUM(fat) and you found an optimal solution with SUM(fat) = 100.
        To generate one more solution you change the problem by adding a new constraint SUM(fat) <= 99.99999.
        Say that now you get an optimal solution with SUM(fat) = 96. You now change the previous constraint into
        SUM(fat) <= 95.99999. And so on, until the problem becomes infeasible.
        """
		packages = set()

		initial_threshold = 1e-10

		if self.query.objective["type"]=="minimize":
			sense = "G"
			oper = operator.add
		elif self.query.objective["type"]=="maximize":
			sense = "L"
			oper = operator.sub
		else:
			raise Exception()

		threshold = initial_threshold

		# Create copy of the problem
		problem = copy.copy(self)

		# Add a dumb constraint that will be changed later
		extra_constr_name = "new-optimal-sol-constr"
		problem.add_linear_constraint([1.0] * len(self.tuple_variables), "G", 0.0, extra_constr_name)

		while True:
			# Try to get optimal solution from current problem
			try:
				optimal_package, cplex_run_info = problem.get_optimal_package(packages)

			except InfeasiblePackageQuery:
				debug('++++++ Infeasible')
				break

			except ILPPackageIgnored:
				debug('++++++ Ignored')
				# Increase the threshold by a little amount
				threshold *= 2

			else:
				debug('++++++ Optimal')
				packages.add(optimal_package)
				yield optimal_package
				threshold = initial_threshold

			optimal_score = optimal_package.get_objective_value()

			problem.remove_linear_constraint(extra_constr_name)
			problem.add_linear_constraint(
				problem.objective_weights, sense, oper(optimal_score, threshold),
				extra_constr_name)
