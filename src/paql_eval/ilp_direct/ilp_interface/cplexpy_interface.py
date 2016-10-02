import time
from itertools import chain, repeat

import cplex
from cplex import SparsePair

from src.data_model.memory_representation.sequence_based import SequenceBasedPackage
from src.paql_eval.exceptions import TimeLimitElapsed, Interrupted
from src.paql_eval.ilp_direct.ilp_interface.cplex_settings import set_cplex_parameters
from src.paql_eval.ilp_direct.ilp_interface.ilp_solver import *
from src.utils.log import verbose_log, log
from src.utils.utils import op_to_cplex_sense, str_u_gc, iter_chunks



def convert_cplex_val(val):
    return cplex.infinity if val == float("inf") else -cplex.infinity if val == -float("inf") else val





class ILPPackageProblemCplexPy(ILPPackageProblem):
    """
    Uses CPLEX's Python API to generate the linear problem corresponding to a given paql problem and solve it.
    """

    def __init__(self, search, **kwargs):
        super(ILPPackageProblemCplexPy, self).__init__(search, **kwargs)
        self.cplex_feasopt_active = False
        self.cplex_conflict_refiner_active = False


    def init_ilp(self):
        super(ILPPackageProblemCplexPy, self).init_ilp()
        c = cplex.Cplex()
        c.set_problem_name("paql-problem")
        self.problem = c


    @property
    def problem_type(self):
        return cplex.Cplex.problem_type[self.problem.get_problem_type()]


    def add_variables(self, n_variables, lb=None, ub=None, var_type=None):
        """
        Add a set of n_variables variables all with same lower bounds, upper bounds and variable types.
        """
        curr_n_vars = self.problem.variables.get_num()

        lb = convert_cplex_val(lb)
        ub = convert_cplex_val(ub)

        if var_type.lower() == "real" or var_type.lower() == "continuous":
            vtype = cplex.Cplex.variables.type.continuous

        elif var_type.lower() == "int" or var_type.lower() == "integer":
            vtype = cplex.Cplex.variables.type.integer

        elif var_type.lower() == "binary" or var_type.lower() == "bool" or var_type.lower() == "boolean":
            vtype = cplex.Cplex.variables.type.binary

        elif var_type.lower() == "auto" or var_type is None:
            vtype = cplex.Cplex.variables.type.binary

        else:
            raise Exception("Vartype '{}' unsupported.".format(var_type))

        if lb is not None and ub is not None:
            self.problem.variables.add(
                lb=[ lb ] * n_variables,
                ub=[ ub ] * n_variables,
                types=[ vtype ] * n_variables)

        elif lb is not None:
            self.problem.variables.add(
                lb=[ lb ] * n_variables,
                types=[ vtype ] * n_variables)

        elif ub is not None:
            self.problem.variables.add(
                ub=[ ub ] * n_variables,
                types=[ vtype ] * n_variables)

        else:
            self.problem.variables.add(
                types=[ vtype ] * n_variables)

        # Return the 0-based indexes of the new variables
        new_var_idxs = xrange(curr_n_vars, curr_n_vars + n_variables)
        return new_var_idxs


    def set_objective_sense(self, sense=None):
        if isinstance(sense, ObjectiveSenseMIN):
            self.problem.objective.set_sense(cplex.Cplex.objective.sense.minimize)

        elif isinstance(sense, ObjectiveSenseMAX):
            self.problem.objective.set_sense(cplex.Cplex.objective.sense.maximize)

        else:
            raise Exception


    def get_objective_sense(self):
        if self.problem.objective.get_sense() == cplex.Cplex.objective.sense.minimize:
            return "minimize"

        elif self.problem.objective.get_sense() == cplex.Cplex.objective.sense.maximize:
            return "maximize"

        else:
            raise Exception


    def set_linear_objective(self, variables, coefficients):
        self.problem.objective.set_linear(izip(variables, coefficients))


    def get_linear_problem_copy(self):
        return cplex.Cplex(self.problem)


    def get_tuple_solution_values(self):
        # return [ float(v) for v in self.problem.solution.get_values() ]
        return self.problem.solution.get_values(list(self.tuple_variables))


    def add_linear_constraints(self, linear_constraints):
        """
        Add multiple constraints in one shot
        NOTE: linear_constraints is a list/iterable of LinearConstraint objects
        """
        print "Adding constraints to CPLEX..."

        # Add groups of constraints together
        add_together_n = 100000  # 100000  # 10000
        added = 0
        for constraints in iter_chunks(add_together_n, linear_constraints):
            print "Adding batch of {} linear constraints to CPLEX problem...".format(len(constraints))

            lin_expr = [None] * len(constraints)
            senses = [None] * len(constraints)
            rhs = [None] * len(constraints)
            names = [None] * len(constraints)
            for i, constraint in enumerate(constraints):
                sparse_pair = SparsePair(ind=constraint.get_variables(), val=constraint.get_coefficients())
                lin_expr[i] = sparse_pair
                senses[i] = op_to_cplex_sense(constraint.op)
                rhs[i] = convert_cplex_val(constraint.rhs)
                names[i] = constraint.get_name()

            self.problem.linear_constraints.add(
                lin_expr=lin_expr,
                senses=senses,
                rhs=rhs,
                names=names)

            added += len(constraints)

        print "Adding constraints to CPLEX: done. " \
              "Added {} new constraints. " \
              "Problem now contains {} constraints.".format(added, self.problem.linear_constraints.get_num())


    def remove_linear_constraints(self, removing_linear_constraints):
        # Find the id's of the constraints to be removed; remove constraints from this class object.
        removing_ids = []
        # NOTE: Using list(removing_linear_constraints) in case removing_linear_constraints = self.linear_constraints
        for c in list(removing_linear_constraints):
            # TODO: Not very efficient with lots of constraints
            c_index = [ i for i, x in enumerate(self.linear_constraints) if x.cid == c.cid ]
            if c_index:
                assert len(c_index) == 1
                del self.linear_constraints[c_index[0]]
                # c_indexes.append(c_index[0])
                self.removed_linear_constraints.append(c)
                # print "POST:", c
                removing_ids.append(c.cid)

        # Remove constraints from Cplex object
        self.problem.linear_constraints.delete(removing_ids)


    def min_error_removed_linear_constraints(self):
        """
        Minimize the error of the previously removed linear constraints.

        Assuming that you have already removed some linear constraints, minimize the error of their
        removal inteded as the sum of absolute distances between the linear constraints and their
        right-hand sides.

        For instance, if you removed SUM(protein) <= 5, then this is added to the objective:
            min <existing-objective> + |SUM(protein) - 5|

        If the objective type was max, the objective becomes the following:
            max <existing-objective> - |SUM(protein) - 5|
        """
        n_e_vars = len(self.removed_linear_constraints) * 2

        # Add a pair of (continuous) variables e+ >= 0 and e- >= 0, for each (removed) conflicting constraint
        eplus_vars = self.add_variables(n_variables=n_e_vars / 2, lb=0, var_type="continuous")
        eminus_vars = self.add_variables(n_variables=n_e_vars / 2, lb=0, var_type="continuous")

        print self.n_tuple_variables
        print len(eplus_vars)
        print len(eminus_vars)
        assert isinstance(self.problem, cplex.Cplex)
        print "n binaries", self.problem.variables.get_num_binary()
        print "n all", self.problem.variables.get_num()
        print "n integers", self.problem.variables.get_num_integer()

        # Set objective coefficients of e variables all to 1 (if minimization, otherwise -1)
        if self.problem.objective.get_sense() == cplex.Cplex.objective.sense.minimize:
            self.problem.objective.set_linear(izip(chain(eplus_vars, eminus_vars), repeat(1, n_e_vars)))
        else:
            self.problem.objective.set_linear(izip(chain(eplus_vars, eminus_vars), repeat(-1, n_e_vars)))

        adding_constraints = list()

        # For minimizing error in SUM(attr) for each attr in the query package
        for i, lc in enumerate(self.removed_linear_constraints):
            def get_coeff_function(_ugc):
                yield 1
                yield -1
                for coeff in self.get_aggregate_constraint_coefficients(_ugc.aggr, _ugc.attr):
                    yield coeff

            def get_vars_function(_i):
                yield eplus_vars[_i]
                yield eminus_vars[_i]
                for var in self.tuple_variables:
                    yield var

            lc = LinearConstraint(
                cid=self.new_constraint_id(),
                vals_func=(get_coeff_function, (lc.ugc,)),
                vars_func=(get_vars_function, (i,)),
                op=operator.eq,
                rhs=lc.rhs)

            print "VALS", lc.get_coeff_function
            print "VARS", lc.get_vars_function

            adding_constraints.append(lc)

        self.add_linear_constraints(adding_constraints)


    def get_linear_problem_string(self):
        lp_file_name = self.store_linear_problem_string("/tmp")
        return open(lp_file_name).read()


    def store_linear_problem_string(self, dirpath, filename=None):
        if filename is None:
            lp_file_name = "{}/lpsearch-cplex-{}.lp".format(dirpath, self.n_solver_solve_calls)
        else:
            lp_file_name = "{}/{}".format(dirpath, filename)
        self.problem.write(lp_file_name)
        log("Problem stored in file '{}'".format(lp_file_name))
        return lp_file_name


    def get_optimal_package(self, packages_to_ignore=None, accept_non_optimal_feasible=False, **kwargs):
        self.setup_problem_parameters()

        # Solve with CPLEX
        cplex_run_info = self.solve()

        # If problem was infeasible, you may try feasopt or conflict refiner
        if cplex_run_info.cplex_status == cplex.Cplex.solution.status.MIP_infeasible:
            if self.cplex_feasopt_active:
                cplex_run_info = self.solve(feas_opt=True)

            elif self.cplex_conflict_refiner_active:
                cplex_run_info = self.solve(conflict_refine=True)

            else:
                raise InfeasiblePackageQuery

        #######################################################################
        # Check CPLEX outcome and return solution (if problem was feasible)
        #######################################################################
        if cplex_run_info.cplex_status == cplex.Cplex.solution.status.MIP_optimal \
                or self.problem.solution.get_status() == cplex.Cplex.solution.status.optimal_tolerance:
            combo = [i + 1 for i, v in enumerate(self.get_tuple_solution_values()) if round(v)==1]

        elif cplex_run_info.cplex_status == cplex.Cplex.solution.status.MIP_optimal_relaxed_sum \
                or self.problem.solution.get_status() == cplex.Cplex.solution.status.MIP_feasible_relaxed_sum \
                or self.problem.solution.get_status() == cplex.Cplex.solution.status.MIP_optimal_relaxed_quad \
                or self.problem.solution.get_status() == cplex.Cplex.solution.status.MIP_feasible:
            combo = [i + 1 for i, v in enumerate(self.get_tuple_solution_values()) if round(v)==1]

        elif cplex_run_info.cplex_status == cplex.Cplex.solution.status.MIP_time_limit_infeasible:
            raise TimeLimitElapsed(self.problem.parameters.timelimit.get())

        elif self.problem.solution.get_status() == cplex.Cplex.solution.status.MIP_time_limit_feasible:
            if accept_non_optimal_feasible:
                combo = [i + 1 for i, v in enumerate(self.get_tuple_solution_values()) if round(v)==1]
            else:
                raise TimeLimitElapsed(self.problem.parameters.timelimit.get())

        elif cplex_run_info.cplex_status == cplex.Cplex.solution.status.MIP_abort_infeasible:
            raise Interrupted

        elif cplex_run_info.cplex_status == cplex.Cplex.solution.status.MIP_abort_feasible:
            if accept_non_optimal_feasible:
                combo = [i + 1 for i, v in enumerate(self.get_tuple_solution_values()) if round(v)==1]
            else:
                raise Interrupted

        else:
            raise Exception("Cplex status unsupported: {}".format(self.problem.solution.get_status()))

        # Return package (if not in packages to ignore)
        package = SequenceBasedPackage(self.search, combo)
        if packages_to_ignore is None or package not in packages_to_ignore:
            return package, cplex_run_info
        else:
            # ignore this package
            raise ILPPackageIgnored


    def get_one_feasible_package(self, packages_to_ignore):
        self.setup_problem_parameters()

        # Remove objective function (don't need it for the task of generating feasible packages)
        self.problem.objective.set_linear([ (v, 0) for v in self.tuple_variables ])

        # Solve
        package, cplex_run_info = self.solve()
        if packages_to_ignore is None or package not in packages_to_ignore:
            return package, cplex_run_info
        else:
            # ignore this package
            raise ILPPackageIgnored


    def setup_problem_parameters(self):
        verbose_log("Setting CPLEX problem params")

        # Set time limit for solver
        timelimit_sec = self.search.get_remaining_time()
        if timelimit_sec is not None:
            self.problem.parameters.timelimit.set(timelimit_sec)

        # Set CPLEX parameters
        set_cplex_parameters(self.problem)


    def compute_conflicting_linear_constraints(self):
        verbose_log("refining...")
        self.problem.conflict.refine(self.problem.conflict.linear_constraints())
        verbose_log("done.")

        print "UGC's:"
        for i, ugc in enumerate(self.search.query.uncoalesced_gcs):
            print i, "GC:", str_u_gc(ugc)

        if self.problem.solution.get_status() == cplex.Cplex.solution.status.conflict_minimal:

            conflict_statuses = self.problem.conflict.get()
            verbose_log("CONFLICT STATUSES:", conflict_statuses)

            selected_group_ids = [
                i for i in xrange(len(conflict_statuses))
                if conflict_statuses[i] == self.problem.conflict.group_status.member
                or conflict_statuses[i] == self.problem.conflict.group_status.possible_member
            ]
            verbose_log("SELECTED GROUP IDS:", selected_group_ids)

            conflict_groups = self.problem.conflict.get_groups(selected_group_ids)
            verbose_log("CONFLICT GROUPS:", conflict_groups)

            self.conflicting_linear_constraints = list()
            print "INCONSISTENT GCs:"
            for preference, constrs in conflict_groups:
                # The preference for now is unused (it is possible to assign a weight to each constraint)
                assert preference == 1.0
                for constr_type, constr_id in constrs:
                    verbose_log("CONSTR ID:", constr_id)
                    if constr_type == cplex.Cplex.conflict.constraint_type.linear:
                        verbose_log("LINEAR RHS:", self.problem.linear_constraints.get_rhs(constr_id))

                        ugc = self.search.query.uncoalesced_gcs[constr_id]
                        print constr_id, str_u_gc(ugc)

                        linear_constraint = filter(lambda x: x.cid == constr_id, self.linear_constraints)[0]

                        self.conflicting_linear_constraints.append(linear_constraint)
                    elif constr_type == cplex.Cplex.conflict.constraint_type.upper_bound:
                        raise Exception
                    else:
                        raise Exception

            if len(self.conflicting_linear_constraints) == 0:
                raise Exception("Problem was infeasible but didn't find any problematic linear constraint.")

            verbose_log("minimial_infeasible_constrs:", self.conflicting_linear_constraints)

        else:
            raise Exception("Cplex status unsupported: {}".format(self.problem.solution.get_status()))


    def solve(self, feas_opt=False, conflict_refine=False):
        if self.store_all_solved_problems:
            self.store_linear_problem_string("/tmp")

        # Initialize running time counters
        cplex_waltime = -self.problem.get_time()
        cplex_dettime = -self.problem.get_dettime()
        wallclock_time = -time.time()
        cputicks_time = -time.clock()

        # Solve the problem with CPLEX
        if not feas_opt and not conflict_refine:
            # Just use the standard CPLEX solve() procedure
            verbose_log("Running CPLEX solve()...")
            self.problem.solve()

        # Solve with CPLEX Feasopt
        elif feas_opt:
            # Mode 0 (default): Minimize the sum of all required relaxations in first phase only
            # Mode 1: Minimize the sum of all required relaxations in first phase and execute second
            #         phase to find optimum among minimal relaxations
            # Mode 2: Minimize the number of constraints and bounds requiring relaxation in first phase only
            # Mode 3: Minimize the number of constraints and bounds requiring relaxation in first phase and
            #         execute second phase to find optimum among minimal relaxations
            # Mode 4: Minimize the sum of squares of required relaxations in first phase only
            # Mode 5: Minimize the sum of squares of required relaxations in first phase and execute second
            #         phase to find optimum among minimal relaxations
            self.problem.parameters.feasopt.mode.set(0)
            verbose_log("Running CPLEX feasopt(...)...")
            self.problem.feasopt(self.problem.feasopt.linear_constraints())

        # Solve after removing constraints with Conflict Refine
        elif conflict_refine:
            self.remove_conflicting_global_constraints()
            verbose_log("Running CPLEX solve()...")
            self.problem.solve()

        self.n_solver_solve_calls += 1

        # Update running time counters
        cplex_waltime += self.problem.get_time()
        cplex_dettime += self.problem.get_dettime()
        wallclock_time += time.time()
        cputicks_time += time.clock()

        cplex_run_info = CPLEXRunInfo(
            cplex_problem_size=self.problem.variables.get_num(),
            cplex_n_problem_linear_constraints=self.problem.linear_constraints.get_num(),
            cplex_wallclock_time=cplex_waltime,
            cplex_detticks_time=cplex_dettime,
            sys_wallclock_time=wallclock_time,
            sys_cputicks_time=cputicks_time,
            wallclock_time_to_store=None,
            cputicks_time_to_store=None,
            cplex_feas_opt_used=feas_opt,
            cplex_conflict_refiner_used=self.conflicting_linear_constraints is not None,
            cplex_status = self.problem.solution.get_status())

        return cplex_run_info


    def gen_feasible_packages_via_solution_pool(self):
        # Handle the case in which the base relation is totally empty
        if len(self.tuple_variables) <= 0:
            # If there's no tuple in the base table, the only candidate is the empty package
            empty_package = SequenceBasedPackage(self.search, tuple())
            if empty_package.is_valid():
                yield empty_package
                return
            else:
                raise InfeasiblePackageQuery

        if not cplex.Cplex.problem_type[self.problem.get_problem_type()]=="MILP":
            raise NotILPError("Not a MILP problem.")

        self.problem.parameters.mip.pool.absgap.set(0.0)
        self.problem.parameters.mip.pool.intensity.set(4)
        self.problem.parameters.mip.limits.populate.set(2100000000)
        self.problem.parameters.emphasis.mip.set(1)
        self.problem.populate_solution_pool()

        # Check cplex_solver outcome
        if self.problem.solution.get_status()==cplex.Cplex.solution.status.MIP_infeasible:
            raise InfeasiblePackageQuery

        # TODO: The second case here should be checked. Why does it happen?
        elif self.problem.solution.get_status()==cplex.Cplex.solution.status.optimal_populated_tolerance\
                or self.problem.solution.get_status()==cplex.Cplex.solution.status.MIP_optimal:
            for solution_name in self.problem.solution.pool.get_names():
                combo = [
                    i + 1
                    for i, v in enumerate(self.problem.solution.pool.get_values(solution_name))
                    if round(v)==1
                ]
                package = SequenceBasedPackage(self.search, combo)
                yield package

        else:
            raise Exception("Cplex status unsupported: {}".format(self.problem.solution.get_status()))



class CPLEXRunInfo:
    def __init__(self, cplex_problem_size, cplex_n_problem_linear_constraints,
                 cplex_wallclock_time, cplex_detticks_time,
                 sys_wallclock_time, sys_cputicks_time,
                 wallclock_time_to_store, cputicks_time_to_store,
                 cplex_conflict_refiner_used, cplex_feas_opt_used,
                 cplex_status):
        self.cplex_wallclock_time = cplex_wallclock_time
        self.cplex_detticks_time = cplex_detticks_time
        self.cplex_problem_size = cplex_problem_size
        self.cplex_n_problem_linear_constraints = cplex_n_problem_linear_constraints
        self.wallclock_time_to_store = wallclock_time_to_store
        self.cputicks_time_to_store = cputicks_time_to_store
        self.sys_wallclock_time = sys_wallclock_time
        self.sys_cputicks_time = sys_cputicks_time
        self.cplex_conflict_refiner_used = cplex_conflict_refiner_used
        self.cplex_feas_opt_used = cplex_feas_opt_used
        self.cplex_status = cplex_status

    def strlist(self):
        return [
            "* CPLEX Run Info:",
            "  CPLEX wall-clock time: {}".format(self.cplex_wallclock_time),
            "  CPLEX det-ticks time: {}".format(self.cplex_detticks_time),
            "  CPLEX problem size: {}".format(self.cplex_problem_size),
        ]

    def __str__(self):
        return "\n".join(self.strlist())
