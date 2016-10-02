import sys
from itertools import izip

import cplex
import operator

import time

from src.paql.aggregates import CountAggr, SumAggr, AvgAggr, MinAggr, MaxAggr
from src.paql.constraints import UGlobalConstraint
from src.paql_eval.exceptions import TimeLimitElapsed
from src.paql_eval.ilp_direct.ilp_interface.cplex_settings import set_cplex_parameters
from src.paql_eval.ilp_direct.ilp_interface.cplexpy_interface import CPLEXRunInfo, ILPPackageProblemCplexPy,\
    convert_cplex_val
from src.paql_eval.ilp_direct.ilp_interface.ilp_solver import LinearConstraint
from src.utils.log import debug
from src.utils.utils import op_to_cplex_sense



class CplexInterface(object):

    def __init__(self, search, store_lp_problems_dir):
        self.search = search
        self.store_lp_problems_dir = store_lp_problems_dir


    def _cplex_augment(self, cid, cid_orig_space_tuples, cid_basis_sol, empty_cids_representatives,
                       count_basis, sums_basis, try_solve_infeasible=False):
        assert cid is not None or len(cid_orig_space_tuples) == 0

        print "Creating augmenting problem on cluster {} ({} orig tuples, {} reprs) with CPLEX...".format(
            cid, len(cid_orig_space_tuples), len(empty_cids_representatives))

        c = self._create_cplex_augmenting_problem(
            "augmenting-problem__spaceid_{}_cid_{}".format(self.search.this_spaceid, cid),
            cid_orig_space_tuples, cid_basis_sol, empty_cids_representatives, count_basis, sums_basis)

        print "created."
        sys.stdout.flush()

        # True unless some trials are infeasible
        is_cid_feasible = True

        # Try to solve CPLEX problem until problem is feasible
        while True:
            # TRY TO SOLVE AUGMENTING PROBLEM
            print "Solving..."
            sys.stdout.flush()
            self._cplex_solve(c)
            print "solved."
            sys.stdout.flush()

            # =========================================== #
            # AUGMENTING PROBLEM FAILURE                  #
            # =========================================== #
            # If problem is infeasible, return a minimal set of infeasible constraints
            if c.solution.get_status() == cplex.Cplex.solution.status.MIP_infeasible:
                debug("Problem is INFEASIBLE.")

                is_cid_feasible = False

                #################################################
                # Try to augment this cluster (e.g. by modifying the sketch_refine)
                #################################################
                if try_solve_infeasible:
                    minimial_infeasible_constrs = \
                        Cplex_Interface.get_minimial_infeasible_constraints(c, self.search.query)

                    # Identify problematic constraints and their corresponding attributes
                    problematic_attrs = sorted([m[2][0][1] for m in minimial_infeasible_constrs if m[2][0][1]!="*"])

                    ##########################################################
                    # Project the sketch_refine onto fewer (problematic) dimensions
                    ##########################################################
                    if problematic_attrs!=self.search.clust_attrs:
                        self.search.project_clustering(problematic_attrs)

                    ##########################################################
                    # Abstract the sketch_refine
                    ##########################################################
                    else:
                        self.search.abstract_clustering()

                    # Keep re-trying this augmenting problem (try to go down the paql_eval tree, don't give up!)
                    return "REDO", None

                elif False:
                    print "FeasOpt..."
                    c.parameters.feasopt.mode.set(1)
                    c.feasopt(c.feasopt.linear_constraints())
                    print "done"
                    n_vars = len(cid_orig_space_tuples) + len(empty_cids_representatives)
                    sol_values = c.solution.get_values(list(xrange(n_vars)))

                    return is_cid_feasible, sol_values

                else:
                    return is_cid_feasible, None

            # =========================================== #
            # AUGMENTING PROBLEM SUCCESS                  #
            # =========================================== #
            # If problem is feasible, return the solution values
            elif c.solution.get_status() == cplex.Cplex.solution.status.MIP_optimal or \
                            c.solution.get_status() == cplex.Cplex.solution.status.optimal_tolerance:
                n_vars = len(cid_orig_space_tuples) + len(empty_cids_representatives)
                sol_values = c.solution.get_values(list(xrange(n_vars)))

                print "FEASIBLE: RETURNING SOL VALUES"

                return is_cid_feasible, sol_values

            elif c.solution.get_status() == cplex.Cplex.solution.status.MIP_time_limit_infeasible or \
                            c.solution.get_status() == cplex.Cplex.solution.status.MIP_time_limit_feasible:

                raise TimeLimitElapsed

            else:
                raise Exception("CPLEX solution status not supported: '{}'".format(c.solution.get_status()))


    def _create_cplex_augmenting_problem(self,
                                         prob_name,
                                         cluster_orig_tuples,
                                         cluster_basis_sol,
                                         other_clusters_reduced_tuples,
                                         count_basis, sums_basis):
        # Number of original tuples
        N = len(cluster_orig_tuples)

        # Number of representative tuples
        R = len(other_clusters_reduced_tuples)

        # The problem will have N + R tuple variables (number of original tuples + number of representative tuples)

        ################################################################################################################
        # Set variable bounds and types
        ################################################################################################################
        # Variables corresponding to actual tuples are binary
        # Variables corresponding to representative tuples are integer (or continuous) in [0, size-of-corr.-cluster]
        var_types = \
            [cplex.Cplex.variables.type.binary] * N + \
            [self.search.vartype_repr] * R

        var_bounds = \
            [(0.0, 1.0)] * N + \
            [(0.0, self.search.n_tuples_per_cid[r.cid] / self.search.n_repres_per_cid[r.cid])
             for r in other_clusters_reduced_tuples]

        ################################################################################################################
        # Set the objective function
        ################################################################################################################
        # Set the objective sense
        obj_sense = str(self.search.query.objective.sense)

        # Set the objective coefficients
        obj_aggr = self.search.query.objective.get_aggregate()
        if isinstance(obj_aggr, CountAggr):
            obj_coeffs = [1] * (N + R)

        elif isinstance(obj_aggr, SumAggr):
            # Add coefficients for actual tuples from cluster cid
            attr = obj_aggr.args[0]
            obj_coeffs = []
            obj_coeffs.extend(getattr(t, attr) for t in cluster_orig_tuples)
            obj_coeffs.extend(getattr(r, attr) for r in other_clusters_reduced_tuples)

        else:
            raise Exception("Objective function not supported: {}".format(obj_aggr))

        # Init ILP
        ilpp = ILPPackageProblemCplexPy(self.search)
        ilpp.init_ilp()

        # Set tuple variables
        assert len(var_types) == len(var_bounds)
        tuple_vars = ilpp.add_variables(N, 0.0, 1.0, var_type="integer")
        repr_vars = []
        if self.search.vartype_repr == cplex.Cplex.variables.type.integer:
            repr_vartype = "integer"
        elif cplex.Cplex.variables.type.continuous:
            repr_vartype = "real"
        else:
            raise Exception
        for r in other_clusters_reduced_tuples:
            ub_var = self.search.n_tuples_per_cid[r.cid] / self.search.n_repres_per_cid[r.cid]
            repr_vars.append(ilpp.add_variables(1, 0.0, ub_var, var_type=repr_vartype))

        # Fix tuple vars and coefficients
        ilpp.tuple_variables = xrange(len(tuple_vars) + len(repr_vars))

        print "N + R =", N + R, "ilpp.problem.variables.get_num() =", ilpp.problem.variables.get_num()
        assert len(ilpp.tuple_variables) == N + R == ilpp.problem.variables.get_num()

        linear_constraints = []

        if not self.search.query.uncoalesced_gcs:
            # Add dumb constraint
            linear_constraints.append(LinearConstraint(
                cid=len(linear_constraints),
                vals_func=(lambda: [1.0] * (N + R), ()),
                vars_func=(lambda: ilpp.tuple_variables, ()),
                op=operator.ge,
                rhs=0.0))

        for ugc_i, ugc in enumerate(self.search.query.uncoalesced_gcs):
            isinstance(ugc, UGlobalConstraint)
            aggrs = list(ugc.iter_aggregates())
            assert len(aggrs) == 1, aggrs
            aggr = aggrs[0]
            attr = aggr.args[0]

            if isinstance(aggr, CountAggr):
                linear_constraints.append(LinearConstraint(
                    cid=len(linear_constraints),
                    vals_func=(lambda: [1.0] * (N + R), ()),
                    vars_func=(lambda: ilpp.tuple_variables, ()),
                    op=ugc.op,
                    rhs=ugc.rhs - count_basis))

            elif isinstance(aggr, SumAggr):
                # Add coefficients for actual tuples from cluster cid
                linear_constraints.append(LinearConstraint(
                    cid=len(linear_constraints),
                    vals_func=(lambda x, y, _attr:
                               [getattr(t, _attr) for t in x] +
                               [getattr(r, _attr) for r in y],
                               (cluster_orig_tuples, other_clusters_reduced_tuples, attr)),
                    vars_func=(lambda: ilpp.tuple_variables, ()),
                    op=ugc.op,
                    rhs=ugc.rhs - sums_basis[attr]))

            elif isinstance(aggr, AvgAggr):
                # Add coefficients for actual tuples from cluster cid
                avg_basis = (sums_basis[attr] / float(count_basis)) if count_basis != 0 else 0.0
                linear_constraints.append(LinearConstraint(
                    cid=len(linear_constraints),
                    vals_func=(lambda x, y, z, _attr:
                               [getattr(t, _attr) - z for t in x] +
                               [getattr(r, _attr) - z for r in y],
                               (cluster_orig_tuples, other_clusters_reduced_tuples, ugc.rhs, attr)),
                    vars_func=(lambda: ilpp.tuple_variables, ()),
                    op=ugc.op,
                    rhs=0 - avg_basis))

            else:
                raise Exception("This aggregator is not supported yet: %s" % aggr)


        ################################################################################################################
        # Return CPLEX problem
        ################################################################################################################
        cplex_interface = Cplex_Interface(c=ilpp.problem)
        c = cplex_interface.create_cplex_problem2(prob_name, N + R, linear_constraints, obj_sense, obj_coeffs)
        return c


    def _cplex_solve(self, c):
        if not cplex.Cplex.problem_type[c.get_problem_type()] == "MILP":
            raise Exception("Not a MILP problem.")

        # Save CPLEX problem file
        if self.store_lp_problems_dir is not None:
            wallclock_time_to_store = -time.time()
            cputicks_time_to_store = -time.clock()
            problem_file_name = "{}/{}_cplex-{}.lp".format(
                self.store_lp_problems_dir,
                c.get_problem_name(),
                Cplex_Interface.prob_num)
            save_linear_problem_string(c, problem_file_name)
            wallclock_time_to_store += time.time()
            cputicks_time_to_store += time.clock()
        else:
            wallclock_time_to_store = cputicks_time_to_store = 0

        # Initialize running time counters
        cplex_time = -c.get_time()
        cplex_dettime = -c.get_dettime()
        wallclock_time = -time.time()
        cputicks_time = -time.clock()

        # Solve problem
        timelimit_sec = self.search.get_remaining_time()
        if timelimit_sec is not None:
            c.parameters.timelimit.set(timelimit_sec)
        c.solve()

        # Update running time counters
        cplex_time += c.get_time()
        cplex_dettime += c.get_dettime()
        wallclock_time += time.time()
        cputicks_time += time.clock()

        # Update RunInfo
        cplex_run_stats = CPLEXRunInfo(
            cplex_problem_size=c.variables.get_num(),
            cplex_n_problem_linear_constraints=c.linear_constraints.get_num(),
            cplex_wallclock_time=cplex_time,
            cplex_detticks_time=cplex_dettime,
            sys_wallclock_time=wallclock_time,
            sys_cputicks_time=cputicks_time,
            wallclock_time_to_store=wallclock_time_to_store,
            cputicks_time_to_store=cputicks_time_to_store,
            cplex_conflict_refiner_used=False,
            cplex_feas_opt_used=False,
            cplex_status=c.solution.status)
        self.search.current_run_info.strategy_run_info.cplex_run_info.append(cplex_run_stats)

        print "done. Problem size was {}, running time was {}".format(
            cplex_run_stats.cplex_problem_size, cplex_run_stats.sys_wallclock_time)



def get_linear_problem_string(c):
    filepath = "/tmp/lpex.lp"
    save_linear_problem_string(c, filepath)
    return open(filepath).read()



def save_linear_problem_string(c, filepath):
    c.write(filepath)



class Cplex_Interface(object):
    prob_num = 0


    def __init__(self, c):
        assert isinstance(c, cplex.Cplex)
        self.c = c

    def add_linear_constraints2(self, constraints):
        lin_expr = [None] * len(constraints)
        senses = [None] * len(constraints)
        rhs = [None] * len(constraints)
        names = [None] * len(constraints)
        for i, constraint in enumerate(constraints):
            sparse_pair = cplex.SparsePair(ind=constraint.get_variables(), val=constraint.get_coefficients())
            lin_expr[i] = sparse_pair
            senses[i] = op_to_cplex_sense(constraint.op)
            rhs[i] = convert_cplex_val(constraint.rhs)
            names[i] = constraint.get_name()
        self.c.linear_constraints.add(
            lin_expr=lin_expr,
            senses=senses,
            rhs=rhs,
            names=names)


    def create_cplex_problem2(self, probl_name, n_vars, linear_constraints, obj_sense, obj_coeffs):
        assert n_vars == len(obj_coeffs)

        Cplex_Interface.prob_num += 1

        # Create CPLEX problem instance
        c = self.c
        c.set_problem_name(probl_name)

        set_cplex_parameters(c)

        # Set the objective sense
        if obj_sense == "minimize":
            c.objective.set_sense(cplex.Cplex.objective.sense.minimize)

        elif obj_sense == "maximize":
            c.objective.set_sense(cplex.Cplex.objective.sense.maximize)

        else:
            raise Exception("paql query doesn't have an objective.")

        # Set the objective function
        c.objective.set_linear(izip(xrange(n_vars), obj_coeffs))

        # Set the global constraints
        self.add_linear_constraints2(linear_constraints)

        return c


    @staticmethod
    def get_minimial_infeasible_constraints(cprobl, query):
        print "refining..."
        cprobl.conflict.refine(cprobl.conflict.linear_constraints())
        print "done."

        conflict_statuses = cprobl.conflict.get()
        print "CONFLICT STATUSES:", conflict_statuses

        conflict_groups =\
            cprobl.conflict.get_groups([
                                           i for i in xrange(len(conflict_statuses))
                                           if conflict_statuses[i]==cprobl.conflict.group_status.member
                                           or conflict_statuses[i]==cprobl.conflict.group_status.possible_member
                                           ])
        print "CONFLICT GROUPS:", conflict_groups

        minimial_infeasible_constrs = set()
        for pref, constrs in conflict_groups:
            assert pref==1.0
            for constr_type, constr_id in constrs:
                print "CONSTR ID:", constr_id
                if constr_type==cprobl.conflict.constraint_type.linear:
                    print "LINEAR:", cprobl.linear_constraints.get_names(constr_id)
                    minimial_infeasible_constrs.add(
                        ("linear", constr_id, query.uncoalesced_gcs[constr_id])
                    )
                elif constr_type==cprobl.conflict.constraint_type.upper_bound:
                    raise Exception
                    print "UPPER BOUND:", cprobl.tuple_variables.get_upper_bounds(constr_id),
                    minimial_infeasible_constrs.add(
                        ("ub", constr_id, None)
                    )
                else:
                    raise Exception

        if len(minimial_infeasible_constrs)==0:
            raise Exception("Problem was infeasible but didn't find any problematic linear constraint.")

        print "minimial_infeasible_constrs:", minimial_infeasible_constrs

        return minimial_infeasible_constrs
