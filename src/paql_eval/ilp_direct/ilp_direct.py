import inspect
import time

from src.data_model.memory_representation.sequence_based import SequenceBasedPackage
from src.paql_eval.exceptions import LPOverflowError, InfeasiblePackageQuery
from src.paql_eval.ilp_direct.ilp_interface.cplexpy_interface import ILPPackageProblemCplexPy, CPLEXRunInfo
from src.paql_eval.ilp_direct.ilp_interface.ilp_solver import ILPPackageProblem
from src.paql_eval.search import Search, SearchRunInfo, SearchInitInfo
from src.utils.log import verbose_log



class ILPDirectInitInfo(SearchInitInfo):
    def __init__(self):
        super(ILPDirectInitInfo, self).__init__()
        self.ILP_problem_loading_wc_time = None
        self.ILP_problem_loading_ct_time = None
        self.ILP_storing_problem_wc_time = None
        self.ILP_storing_problem_ct_time = None


    def __str__(self):
        res = [
            super(ILPDirectInitInfo, self).__str__(),
            "> ILP Search - Specific Search Run Info:",
            "  ILP loading wc time: {} s".format(self.ILP_problem_loading_wc_time),
        ]
        return "\n".join(res)


    def problem_loading_start(self):
        self.ILP_problem_loading_wc_time = -time.time()
        self.ILP_problem_loading_ct_time = -time.clock()


    def problem_loading_end(self):
        self.ILP_problem_loading_wc_time += time.time()
        self.ILP_problem_loading_ct_time += time.clock()


    def storing_problem_start(self):
        self.ILP_storing_problem_wc_time = -time.time()
        self.ILP_storing_problem_ct_time = -time.clock()


    def storing_problem_end(self):
        self.ILP_storing_problem_wc_time += time.time()
        self.ILP_storing_problem_ct_time += time.clock()



class ILPDirectRunInfo(SearchRunInfo):
    def __init__(self, *args, **kwargs):
        super(ILPDirectRunInfo, self).__init__(*args, **kwargs)
        self.ILP_problem_solving_wc_time = None
        self.ILP_problem_solving_ct_time = None
        self.CPLEX_run_info = None


    def strlist(self):
        s = [
            "* ILP Direct - Specific Search Run Info:",
            "  Max problem size: {} vars".format(self.CPLEX_run_info.cplex_problem_size),
            "  ILP solving wc time: {} s".format(self.ILP_problem_solving_wc_time),
        ]
        for x in self.CPLEX_run_info.strlist():
            s.append("  {}".format(x))
        return s


    def __str__(self):
        s = super(ILPDirectRunInfo, self).strlist()
        for x in self.strlist():
            s.append("  {}".format(x))
        return "\n".join(s)


    def problem_solving_start(self):
        self.ILP_problem_solving_wc_time = -time.time()
        self.ILP_problem_solving_ct_time = -time.clock()


    def problem_solving_end(self):
        self.ILP_problem_solving_wc_time += time.time()
        self.ILP_problem_solving_ct_time += time.clock()



class ILPDirect(Search):
    lp_solver_wallclock_time = None
    lp_solver_cputicks_time = None

    searchruninfo_class = ILPDirectRunInfo
    search_init_info_class = ILPDirectInitInfo


    def __init__(self, store_all_solved_problems=False, *args, **kwargs):
        super(ILPDirect, self).__init__(*args, **kwargs)
        self.ilp_problem = None
        self.store_all_solved_problems = store_all_solved_problems
        # self.store_lp_problems_dir = None


    def enumerate_package_space_hook(self, *args, **kwargs):
        return self.enumerate_feasible_packages_hook(*args, **kwargs)


    @staticmethod
    def nice_name():
        return "ILP Search"


    def init(self, query, store_lp_problems_dir=None, **kwargs):
        """
        Ovverrides init_search of Search class to allow storing LP problems.
        """
        super(ILPDirect, self).init(query, **kwargs)

        # Settings
        ilp_solver_interface = kwargs.get("ilp_solver_interface", ILPPackageProblemCplexPy)
        assert inspect.isclass(ilp_solver_interface) and issubclass(ilp_solver_interface, ILPPackageProblem)
        problem_type = kwargs.get("problem_type", "auto")

        self.ilp_problem = ilp_solver_interface(
            search=self,
            store_all_solved_problems=self.store_all_solved_problems)
        assert isinstance(self.ilp_problem, ILPPackageProblem)

        verbose_log("Loading ILP problem...")
        self.current_init_info.problem_loading_start()
        self.ilp_problem.load_ilp_from_paql(problem_type)
        self.current_init_info.problem_loading_end()
        verbose_log("ILP problem loaded.")

        # Store ILP to a file (if needed)
        if store_lp_problems_dir is not None:
            verbose_log("Saving ILP problem into folder '{}'...".format(store_lp_problems_dir))
            self.current_init_info.storing_problem_start()
            self.ilp_problem.store_linear_problem_string(store_lp_problems_dir)
            self.current_init_info.storing_problem_end()
            verbose_log("ILP problem saved.")


    def get_linear_problem_string(self):
        return self.ilp_problem.get_linear_problem_string()


    def enumerate_feasible_packages_hook(self, *args, **kwargs):
        # FIXME: Make sure that the objective function is not set, you don't need it for this function

        ilp_solver_interface = kwargs.get("ilp_solver_interface")
        assert inspect.isclass(ilp_solver_interface) and issubclass(ilp_solver_interface, ILPPackageProblem)

        strategy = kwargs.get("strategy")
        assert inspect.ismethod(strategy)

        ilp_problem = ilp_solver_interface(self)
        ilp_problem.load_ilp_from_paql()
        package_generator = ilp_problem.strategy()

        try:
            for p in package_generator:
                yield p
        except OverflowError:
            raise LPOverflowError


    def get_optimal_package_hook(self, *args, **kwargs):
        # Settings
        cplex_feasopt = kwargs.get("cplex_feasopt", False)
        cplex_conflict_refine = kwargs.get("cplex_conflict_refine", False)
        # They can't both be set to True
        assert cplex_feasopt is False or cplex_conflict_refine is False

        self.ilp_problem.cplex_feasopt_active = cplex_feasopt
        self.ilp_problem.cplex_conflict_refiner_active = cplex_conflict_refine

        if self.N == 0:
            # If there's no tuple in the base table, the only candidate is the empty package
            empty_package = SequenceBasedPackage(self, tuple())
            if empty_package.is_valid():
                return empty_package
            else:
                raise InfeasiblePackageQuery

        # Solve ILP problem
        verbose_log("Solving ILP problem...")
        self.current_run_info.problem_solving_start()
        optimal_package, cplex_runinfo = self.ilp_problem.get_optimal_package(**kwargs)
        self.current_run_info.problem_solving_end()
        verbose_log("ILP problem solved.")

        # Add running info about ILP solver to the specific running info
        assert isinstance(cplex_runinfo, CPLEXRunInfo)
        self.current_run_info.CPLEX_run_info = cplex_runinfo

        return optimal_package


    def remove_conflicting_global_constraints(self):
        self.ilp_problem.remove_conflicting_global_constraints()


    def remove_all_global_constraints(self):
        self.ilp_problem.remove_all_global_constraints()


    def min_error_removed_global_constraints(self):
        self.ilp_problem.min_error_removed_linear_constraints()


    def get_one_feasible_package_hook(self, *args, **kwargs):
        raise NotImplemented
