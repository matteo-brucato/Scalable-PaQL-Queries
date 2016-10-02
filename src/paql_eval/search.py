from __future__ import division

import time
import traceback
from abc import ABCMeta, abstractmethod
from logging import debug, info

from src.dbms.db import DBConnection, DBCursor
from src.dbms.dbms_settings import data_dbms_settings
from src.data_model.storage_model.simple_table import SimpleTableDataModel
from src.paql.package_query import PackageQuery
from src.paql_eval.cardinality_pruning import get_cardinality_pruning_gain, intersect_cardinality_bounds
from src.paql_eval.exceptions import TimeLimitElapsed, InfeasiblePackageQuery
from src.utils.utils import n_of_subsets



class SearchInitInfo(object):
    def __init__(self):
        # Time taken to init the query (last inited query only)
        self.init_wallclock_time = None
        self.init_cputicks_time = None

    def __str__(self):
        c = [
            "SearchInitInfo:",
            "init_wallclock_time: {}".format(self.init_wallclock_time),
            "init_cputicks_time: {}".format(self.init_cputicks_time),
        ]
        return "\n".join(c)

    def start(self):
        self.init_wallclock_time = -time.time()
        self.init_cputicks_time = -time.clock()

    def end(self):
        self.init_wallclock_time += time.time()
        self.init_cputicks_time += time.clock()



class SearchRunInfo(object):
    def __init__(self, query, timelimit, store_packages=False):
        assert isinstance(query, PackageQuery)
        assert timelimit is None or type(timelimit) is float or type(timelimit) is int
        assert type(store_packages) is bool

        self.query = query
        self.timelimit = timelimit
        self.store_packages = store_packages

        # The time when the last paql_eval was started. The moment any of the paql_eval methods of this class is called,
        # these values are reset.
        self.wallclock_time_started = None
        self.cputicks_time_started = None

        # This is a list of {package, ..., time}, where time is the time to produce each candidate package, starting
        # from the time the paql_eval was issued. If you want to use them to compute time elapsed between two candidates
        # are enumerated, you should take this into account (i.e. subtract the two values).
        self.info_packages = []


    def start(self):
        self.wallclock_time_started = time.time()  # in seconds
        self.cputicks_time_started = time.clock()


    def strlist(self):
        s = [
            "* General Search Run Info:",
            "  Wallclock started = {}".format(self.wallclock_time_started),
            "  Cputicks started = {}".format(self.cputicks_time_started),
        ]
        for t in self.info_packages:
            s.append("  * Package info:")
            s.append("    WC time: " + str(t["wc_time"]))
            s.append("    CT time: " + str(t["ct_time"]))
            if "specific_info" in t:
                s.append("    Specific info: {}".format("\n    ".join(t["specific_info"].strlist())))
        return s


    def __str__(self):
        return "\n".join(self.strlist())


    def invalidate(self):
        # Delete all CandidatePackages that have been associated with this paql_eval info object
        for x in self.info_packages:
            if x["package"] is not None:
                x["package"].invalidate()


    def get_remaining_time(self):
        """
        Returns remaining time to time limit in seconds. Time is wallclock time because we are interested
        in actual interactivity.
        Returns None if there's no time limit set.
        Raises TimeLimitElapsed if time limit has already elapsed.
        """
        if self.timelimit is not None:
            remaining_time = self.wallclock_time_started + self.timelimit - time.time()
            if remaining_time <= 0:
                raise TimeLimitElapsed
            return remaining_time
        else:
            return None


    def new_package_generated(self, package):
        """
        :param package: The CandidatePackage this specific info is reffering to
        :param other_specific_runinfo: Other running time info produced by subclasses of Search, may be None
        """
        # TODO: When using SQL and CPLEX, you should at least count the cpu ticks from their APIs. Perhaps you should
        # also use the CPLEX API to count info about wall clock time in some way.
        wallclock_time = time.time() - self.wallclock_time_started
        cputicks_time = time.clock() - self.cputicks_time_started
        assert wallclock_time > 0
        assert cputicks_time > 0

        # print "=*" * 50
        # print "NEW PACKAGE AT:", time.time()
        # print "=*" * 50

        self.info_packages.append(
            {"package": package if self.store_packages else None,
             "wc_time": wallclock_time,
             "ct_time": cputicks_time})


    def get_last_package(self):
        return self.info_packages[-1]["package"]


    def get_last_package_wallclock_time(self):
        return self.info_packages[-1]["wc_time"]


    def get_last_package_cputicks_time(self):
        return self.info_packages[-1]["ct_time"]



class Search(object):
    """
    A paql_eval object creates a connection (i.e. session) to the underlining database, which is maintained until
    the object gets destroyed (with __del__(), for instance). DB connections can be safely shared among threads,
    therefore the same paql_eval object can be shared among different threads.

    Creating a paql_eval object doesn't perform any paql_eval operation at all. It only connects to the database backend.
    It doesn't even create any underlining data structures in the database.

    The first thing to do with a paql_eval object is to init_search(paql, **kwargs) it, by providing a proper
    PackageQuery object. This is the query which will be evaluated by the paql_eval object. The init procedure will
    simply create all data structures needed by the paql_eval algorithms. Most importantly, it creates temporary
    tables in the database backend, which will be used only by this paql_eval object. These tables only exist for
    the current DB connection, so they are absolutely specific to the paql_eval object being used. If there is another
    paql_eval object active in another thread, it will see different tables (with the same names, but in a different
    name space).

    When you want to release the connection to the DB, you need to call close().
    """

    # This class is abstract, meaning that instances of this class cannot be created -- only of its subclasses.
    __metaclass__ = ABCMeta

    searchruninfo_class = SearchRunInfo
    search_init_info_class = SearchInitInfo


    def __init__(self, existing_search=None, use_connection=None):
        # State of paql_eval
        self._search_initialized = False
        self._search_started = False

        # DB cursor (will be instantiated when paql_eval will be inited with a query)
        self.db = None

        # For queries
        self.package_table = None
        self.lb = self.ub = None
        self.N = None
        self.search_space_size = None
        self.query = None
        self.cardinality_pruning = None
        self.cardinality_pruning_type = None
        self.schema_name = None
        self.table_name = None

        # Run info (for current run and all previous runs of this paql_eval)
        self.current_run_info = None
        self.current_init_info = None
        self.run_infos = None

        self.existing_search = existing_search
        self.use_connection = use_connection

        if self.existing_search is None and self.use_connection is None:
            # Start a new DB connection, which can be used by multiple Search runs withing this object
            self.db_connection = DBConnection(**data_dbms_settings)

        elif self.use_connection is not None:
            # Use the specified DB connection
            self.db_connection = use_connection

        else:
            # Use the DB connection from an existing Search object
            assert isinstance(existing_search, Search),\
                "{} {}".format(existing_search, type(existing_search))
            self.db_connection = self.existing_search.db_connection


    def close(self):
        debug("Deleting {} at {}.".format(self.__class__.__name__, id(self)))

        if self._search_initialized:
            self._end_search()


    def commit(self):
        info("Committing transaction on paql_eval object at {}.".format(id(self)))
        self.db.commit()


    def abort(self):
        info("Aborting transaction on paql_eval object at {}.".format(id(self)))
        self.db.abort()


    def get_remaining_time(self):
        """
        Returns remaining time to time limit in seconds.
        Returns None if there's no time limit set.
        Raises TimeLimitElapsed if time limit has already elapsed.
        """
        assert self._search_started
        return self.current_run_info.get_remaining_time()


    def check_timelimit(self):
        remaining_time = self.get_remaining_time()
        self.db.set_timelimit_sec(remaining_time)


    def get_last_run_info(self):
        return self.run_infos[-1]


    def get_init_info(self):
        return self.current_init_info


    @staticmethod
    @abstractmethod
    def nice_name():
        pass


    def _generic_package_generator(self, timelimit, specific_generator):
        assert self._search_initialized
        assert not self._search_started
        assert timelimit is None or timelimit >= 0
        self._search_started = True

        # This is infeasibility detected with cardinality pruning
        if self.lb is None or self.ub is None:
            self._end_search()
            raise InfeasiblePackageQuery(
                "Infeasibility detected via cardinality pruning, before starting evaluation.")

        # Any SQL query cannot last longer than time limit
        self.db.set_timelimit_sec(timelimit)
        self.current_run_info = self.searchruninfo_class(self.query, timelimit)
        self.run_infos += [ self.current_run_info ]

        self.current_run_info.start()

        # Try generating packages (run search method)
        try:
            for package in specific_generator:
                # Register new generated package
                self.current_run_info.new_package_generated(package)
                yield package

        except Exception as exception:
            print traceback.print_exc()
            self._end_search()
            raise exception

        else:
            # Things to do when the paql_eval procedure has finished successfully
            self._end_search()

        finally:
            # Things to do after the paql_eval finishes either successfully or not
            self.current_run_info = None


    def _end_search(self):
        assert self._search_initialized
        self.commit()
        self._search_started = False
        for rinfo in self.run_infos:
            del rinfo


    def enumerate_package_space(self, timelimit=None, *args, **kwargs):
        info(">>> SEARCHING {}.enumerate_package_space()...".format(self.__class__.__name__))
        package_generator = self.enumerate_package_space_hook(*args, **kwargs)
        for p in self._generic_package_generator(timelimit, package_generator):
            yield p


    @abstractmethod
    def enumerate_package_space_hook(self, *args, **kwargs):
        """
        Subclasses should implement a generator that yields candidate packages (feasible or infeasible), one at a time,
        in the order that the specific method generates them.
        Each method will generate only a portion of the entire paql_eval space.
        Heuristic and approximated method (i.e. suboptimal methods) may not generate some of the feasible packages.
        """
        raise NotImplemented


    def enumerate_feasible_packages(self, timelimit=None, *args, **kwargs):
        info(">>> SEARCHING {}.enumerate_feasible_packages()...".format(self.__class__.__name__))
        package_generator = self.enumerate_feasible_packages_hook(*args, **kwargs)
        for p in self._generic_package_generator(timelimit, package_generator):
            yield p


    @abstractmethod
    def enumerate_feasible_packages_hook(self, *args, **kwargs):
        """
        Subclasses should implement a generator that yields all feasible packages, one at a time, in the order that the
        specific method generates them.
        Some methods may be incomplete, meaning that they may not return some of the feasible pacakges.
        """
        raise NotImplemented


    def get_one_feasible_package(self, timelimit=None, *args, **kwargs):
        assert self.query.objective is None, "get_one_feasible_package() can be called only on non-optimization queries"
        info(">>> SEARCHING {}.get_one_feasible_package()...".format(self.__class__.__name__))

        def package_generator():
            yield self.get_one_feasible_package_hook(*args, **kwargs)

        packages = []
        for p in self._generic_package_generator(timelimit, package_generator()):
            packages.append(p)
        assert len(packages) == 1
        return packages[0]


    @abstractmethod
    def get_one_feasible_package_hook(self, *args, **kwargs):
        """
        Subclasses should implement a method for getting one single feasible package.
        There is no requirement in terms of *which* feasible package to return. More "intelligent" methods
        will try to return a feasible package quickly. Incomplete methods may not find a feasible package
        even if there is one in the paql_eval space.
        """
        raise NotImplemented


    def get_optimal_package(self, timelimit=None, *args, **kwargs):
        print
        print "*" * 50
        print " GET OPTIMAL PACKAGE STARTED"
        print "*" * 50

        assert self.query.objective is not None, "get_optimal_package() can be called only on optimization queries"
        info(">>> SEARCHING {}.get_optimal_package()...".format(self.__class__.__name__))

        def package_generator():
            yield self.get_optimal_package_hook(*args, **kwargs)

        packages = []
        for p in self._generic_package_generator(timelimit, package_generator()):
            packages.append(p)
        assert len(packages) == 1
        return packages[0]


    @abstractmethod
    def get_optimal_package_hook(self, *args, **kwargs):
        """
        Subclasses should implement a method for getting the best feasible package for the objective function.
        Incomplete methods (i.e., methods that do not necessarily see the entire feasible space) may return a
        sub-optimal package, or may return no package at all even if there are packages in the paql_eval space.
        Also,
        """
        raise NotImplemented


    def reset(self):
        # print "RESET..."
        if self._search_initialized:
            self._end_search()
        if self.package_table is not None:
            # print "DESTROY CANDIDATE TABLE"
            self.package_table.destroy()
            self.package_table = None
        # All results from previous runs (especially all generated packages) must be made inaccessible
        if self.run_infos is not None:
            for rinfo in self.run_infos:
                rinfo.invalidate()


    def init(self, query, *args, **kwargs):
        """
        NOTE: Initing a new paql_eval will lose the previous DB cursor, and it will also destroy all running times of
        any previous runs. It is almost like getting a new Search object, except that the connection to the DB is not
        lost. This is convenient if you want to issue multiple queries in parallel: threads can shared the same Search
        object, but all operations will be done in independent environment, and everything will use one single
        connection to the DB.
        """
        self.current_init_info = self.search_init_info_class()
        self.current_init_info.start()

        info(">>> INITIALIZE SEARCH METHOD {}.init_search()...".format(self.__class__.__name__))
        assert isinstance(query, PackageQuery)
        self.reset()
        self._search_initialized = True
        self._search_started = False
        self.query = query
        self.run_infos = []

        # Start a new DB transaction, which will last until the end of the paql_eval procedure
        self.db = DBCursor(
            self.db_connection,
            logfile=None,
            sqlfile=None)

        # Create the package table which will hold the candidate packages
        self.package_table = SimpleTableDataModel(self.db, self.query)

        # NOTE: You should always refer to the core table from now on
        self.schema_name = self.package_table.coretable.schema_name
        self.table_name = self.package_table.coretable.table_name

        # Count tuples from core table (base constraints already applied)
        self.N = self.package_table.get_coretable_size()

        if False:
            self.search_space_size = n_of_subsets(self.N, 0, self.N)
        else:
            self.search_space_size = None

        debug("No of tuples in Core table: %s", self.N)
        debug("Search space size: %s", self.search_space_size)

        # Apply cardinality-based pruning
        self.cardinality_pruning = kwargs.get("cardinality_pruning", False)
        if self.cardinality_pruning:
            # from cardinality_pruning import intersect_cardinality_bounds

            if isinstance(self.cardinality_pruning, str):
                prunetype = self.cardinality_pruning
            elif isinstance(self.cardinality_pruning, bool):
                prunetype = self.cardinality_pruning
            else:
                raise Exception("cardinality pruning type not recognized: {}".format(self.cardinality_pruning))
            self.lb, self.ub = intersect_cardinality_bounds(
                self.db, self.package_table.coretable, self.query.coalesced_gcs, self.N, prunetype)
        else:
            self.lb, self.ub = 0, self.N

        self.commit()
        self.current_init_info.end()


    def get_pruning_gain(self):
        return get_cardinality_pruning_gain(self)
