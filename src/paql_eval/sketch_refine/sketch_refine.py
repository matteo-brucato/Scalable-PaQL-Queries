# coding=utf-8
from __future__ import division

import hashlib
import math
import sys
import time

import cplex
from scipy.optimize import minimize

from src.dbms.utils import sql_table_exists, sql_table_column_exists, sql_index_exists, sql_get_all_indexes
from src.paql.objectives import ObjectiveSenseMAX, ObjectiveSenseMIN
from src.paql_eval.search import Search, SearchRunInfo
from src.paql_eval.sketch_refine.partial_package import PartialPackage
from src.paql_eval.sketch_refine.partitioning.quad_tree import QuadTreePartitioning
from src.utils.log import *
from src.utils.utils import avg



_sr_schema = "sketchrefine"  # "public"

MAX_CLUST_DEPTH = 64
INITIAL_CID = 1
CID_TYPE = "BIGINT"
CID_TYPE_CAST = "{}"



class ReducedSpaceInfeasible(Exception):
    def __init__(self, message, runinfo=None):
        super(ReducedSpaceInfeasible, self).__init__(message)
        self.runinfo = runinfo



class SketchRefineRunInfo(SearchRunInfo):
    def __init__(self, *args, **kwargs):
        super(SketchRefineRunInfo, self).__init__(*args, **kwargs)
        self.strategy_run_info = None


    def __str__(self):
        res = [
            super(SketchRefineRunInfo, self).__str__(),
            "> SketchRefine - Strategy Run Info:",
            str(self.strategy_run_info)
        ]
        return "\n".join(res)



class SketchRefine(Search):
    clust_algo_names = ["quad"]
    spaceid = 0
    searchruninfo_class = SketchRefineRunInfo


    @staticmethod
    def nice_name():
        return "Sketch Refine Search"


    def __init__(self, *args, **kwargs):
        sys.setrecursionlimit(1000000)

        super(SketchRefine, self).__init__(*args, **kwargs)

        self.sr_schema = _sr_schema
        self.mode_marginal = None
        self.this_spaceid = None
        self.store_lp_problems_dir = None
        self.n_tuples_per_cid = None
        self.n_repres_per_cid = None
        self.clust_setting = None
        self.n_clusters = None
        self.clust_size_threshold = None
        self.min_n_clusts = None
        self.epsilon = None
        self.last_clust_size_threshold = None
        self.clust_algo = None
        self.clust_attrs = None
        self.query_attrs = None
        self.dataset_attrs = None
        self.indexing_attrs = None
        self.attrs = None
        self.vartype_repr = None
        self.already_partitioned = None
        self.repr_table_name = None
        self.index_table_name = None
        self.clust_dbms_index_name = None
        self.max_radius = None
        self.max_satisfied_epsilons = None
        self.n_violating_cids = None
        self.worst_violated_epsilons = None
        self.avg_satisfied_epsilons = None
        self.avg_all_epsilons = None
        self.representatives = None
        self.use_representatives = None
        self.clust_attrs_indexes = None
        self.clustering_identifier = None
        self.stored_clust_table_name = None
        self.clustering_online_wc_time = None
        self.clustering_online_ct_time = None
        self.dbms_index_wc_time = None
        self.dbms_index_ct_time = None
        self.repr_wallclock_time = None
        self.repr_cputicks_time = None
        self.recover_clustering_online_wc_time = None
        self.recover_clustering_online_ct_time = None
        self.clustering_recover_w_index_wc_time = None
        self.clustering_recover_w_index_ct_time = None
        self.indexing_wc_time = None
        self.indexing_ct_time = None
        self.use_index = None
        self.cid_type = None
        self.cid_type_cast = None
        self.default_cid = None


    def init(self, query, *args, **kwargs):
        """
        Ovverrides init of Search class
        """
        super(SketchRefine, self).init(query, *args, **kwargs)
        self.last_clust_size_threshold = None
        self.__init(*args, **kwargs)


    def __init(self, dataset_attrs, query_attrs, clust_attrs, clust_algo, clust_setting,
               store_lp_problems_dir, vartype, use_representatives, repartition, use_index, epsilon,
               mode_marginal, already_partitioned, indexing_attrs):
        assert self._search_initialized
        assert not self._search_started
        log("* Initing SketchRefine search method...")

        # Whether marginal sketch_refine is active (when experiment is csize and when data is incrementally clustered)
        self.mode_marginal = mode_marginal

        self.already_partitioned = already_partitioned
        self.this_spaceid = self.spaceid
        SketchRefine.spaceid += 1

        self.dataset_attrs = sorted(dataset_attrs)
        self.query_attrs = sorted(query_attrs)
        self.clust_attrs = sorted(clust_attrs)
        self.indexing_attrs = sorted(indexing_attrs)

        # Clustering algorithm name
        self.clust_algo = clust_algo
        if self.clust_algo not in SketchRefine.clust_algo_names:
            raise Exception("Clustering algorithm name '{}' not recognized. Accepted names are: {}".format(
                self.clust_algo, SketchRefine.clust_algo_names))

        # Quality requirement
        self.epsilon = epsilon if epsilon >= 0 else None

        # Clustering setting
        self.clust_setting = clust_setting

        # Interpret partitioning setting as max cluster size threshold if quad is used
        if self.clust_algo=="quad":
            if "." in self.clust_setting and 0 <= float(self.clust_setting) <= 1:
                self.clust_size_threshold = int(max(1, self.N * float(self.clust_setting)))
                self.min_n_clusts = None
            elif "." not in self.clust_setting:
                self.clust_size_threshold = int(self.clust_setting)
                self.min_n_clusts = None
            elif self.clust_setting.startswith("m"):
                # Interpret setting as "minimum number of clusters"
                self.clust_size_threshold = None
                self.min_n_clusts = int(self.clust_setting[1:])
            else:
                raise Exception("Theshold (clust_setting) should be either a float in [0,1] or an integer")

            print "Cluster Size Threshold set to:", self.clust_size_threshold
            print "Min N of Clusters Threshold set to:", self.min_n_clusts
            print "Expected number of clusters:", self.expected_n_partitions()

        self.store_lp_problems_dir = store_lp_problems_dir

        # Set variable type for representative tuples
        if vartype=="int" or vartype=="integer":
            self.vartype_repr = cplex.Cplex.variables.type.integer
        elif vartype=="continuous" or vartype=="real":
            self.vartype_repr = cplex.Cplex.variables.type.continuous
        else:
            raise Exception("Vartype '{}' unknown.".format(vartype))

        self.use_representatives = use_representatives

        # The attributes to be used to manage both actual tuples and representatives are
        # the union of the sketch_refine attributes and the query attributes
        self.attrs = sorted(set(self.clust_attrs) | set(self.query_attrs))

        # Indexes of the dataset attributes that are actually used for sketch_refine
        self.clust_attrs_indexes = [
            i for i in xrange(len(self.dataset_attrs)) if self.dataset_attrs[i] in self.clust_attrs]

        self.use_index = use_index

        # A unique identifier for this particular partitioning (it depends on the
        # dataset table name, the sketch_refine algorithm, the sketch_refine setting
        # and the sketch_refine attributes being used).
        self.clustering_identifier = hashlib.md5(
            "{schema_name} {data_table_name} {clust_algo} {clust_setting} {epsilon} {clust_attrs}".format(
                schema_name=self.schema_name,
                data_table_name=self.table_name,
                clust_algo=self.clust_algo,
                clust_setting=self.clust_setting,
                epsilon=self.epsilon,
                clust_attrs="_".join("{}".format(self.clust_attrs)),
            )).hexdigest()

        self.repr_table_name = "repr_{}".format(self.clustering_identifier)
        self.stored_clust_table_name = "clus_{}".format(self.clustering_identifier)
        print "Data schema name:", self.schema_name
        print "Data table name:", self.table_name
        print "Repres table name:", self.repr_table_name
        print "Stored clustering table name:", self.stored_clust_table_name

        assert self.use_index and self.indexing_attrs or (not self.use_index and not self.indexing_attrs)

        if self.indexing_attrs:
            self.index_identifier = hashlib.md5(
                "{schema_name} {data_table_name} {clust_algo} {indexing_attrs}".format(
                    schema_name=self.schema_name,
                    data_table_name=self.table_name,
                    clust_algo=self.clust_algo,
                    indexing_attrs="_".join("{}".format(self.indexing_attrs)))).hexdigest()
            self.index_table_name = "idx_{}".format(self.index_identifier)
            print "Index table name:", self.index_table_name

        self.clust_dbms_index_name = "_{}_cid_idx".format(self.table_name)

        # Initialize sketch_refine
        self.clustering_online_wc_time = None
        self.clustering_online_ct_time = None
        self.recover_clustering_online_wc_time = None
        self.recover_clustering_online_ct_time = None

        self.max_clust_depth = int(math.ceil(math.log(
            self.N, len(self.clust_attrs if not use_index else self.indexing_attrs))))

        self.nbits_clust = 1 + (self.max_clust_depth * len(self.clust_attrs if not use_index else self.indexing_attrs))
        log("No clust bits:", self.nbits_clust)

        self.cid_type = CID_TYPE
        self.cid_type_cast = CID_TYPE_CAST.format(self.cid_type, self.nbits_clust)
        self.default_cid = INITIAL_CID

        if self.cid_type.lower()=="bigint" and self.nbits_clust > 8 * 8:
            self.nbits_clust = 8 * 8

        if bool(repartition):
            # Compute and store the clusters
            log("\nClustering, setting = {}, algo = {}, on attributes: {}...".format(
                self.clust_setting, self.clust_algo, ",".join(self.clust_attrs)))
            self.perform_online_partitioning()
            print "Clustering: done."

        elif bool(already_partitioned):
            log("Dataset is already clustered correctly (as claimed by caller): Don't do anything.")
            self.prep_completed_partitioning(store=False)

        else:
            # Recover clusters from DB
            log("\nLoad stored clustering, '{}' on attributes: {}...".format(
                self.clust_algo, ", ".join(self.clust_attrs)))
            self.load_stored_partitioning()
            log("Load stored clustering: done.")

        assert set(self.n_tuples_per_cid.keys())==set(self.n_repres_per_cid.keys()),\
            "Cluster ids differ:\n{}\n{}".format(
                sorted(self.n_tuples_per_cid.keys()),
                sorted(self.n_repres_per_cid.keys()))

        self.check_init()
        self.init_done()


    def check_init(self):
        # Make sure the cluster ID column is present and indexed
        for index_name, col_name, isprimary in sql_get_all_indexes(self.db, self.schema_name, self.table_name):
            if col_name=="cid" and not isprimary:
                break
        else:
            raise Exception("Clustering column 'cid' either not present or not indexed at the DBMS level.")


    def init_done(self):
        print "/-----------------{:-<55}\\".format("")
        print "| Reduce Space initialized{:<47}|".format("")
        print "+-----------------{:-<55}+".format("")
        print "| Query attrs:    {:<55}|".format(",".join(self.query_attrs))
        print "| Data attrs:     {:<55}|".format(",".join(self.dataset_attrs))
        print "| Clust algo:     {:<55}|".format(self.clust_algo)
        print "| Clust setting:  {:<55}|".format(self.clust_setting)
        print "| Repres:         {:<55}|".format(",".join([ur.__name__ for ur in self.use_representatives]))
        print "+-----------------{:-<55}+".format("")
        print "| Clust attrs:    {:<55}|".format(",".join(self.clust_attrs))
        print "| N clusts:       {:<55}|".format(self.n_clusters)
        print "| Min clust size: {:<55}|".format(min(self.n_tuples_per_cid.itervalues()))
        print "| Avg clust size: {:<55}|".format(avg(self.n_tuples_per_cid.itervalues()))
        print "| Max clust size: {:<55}|".format(max(self.n_tuples_per_cid.itervalues()))
        print "+-----------------{:-<55}+".format("")
        print "| Onl. clus time: {:<55}|".format(self.clustering_online_wc_time)
        print "| Idx clust time: {:<55}|".format(self.clustering_recover_w_index_wc_time)
        print "| Idx creat time: {:<55}|".format(self.indexing_wc_time)
        print "\-----------------{:-<55}/".format("")
        print


    def expected_n_partitions(self):
        if self.min_n_clusts is not None:
            return self.min_n_clusts
        else:
            return int(math.ceil(self.N / self.clust_size_threshold))


    def expected_max_partition_size(self):
        if self.clust_size_threshold is not None:
            return self.clust_size_threshold
        else:
            return int(math.ceil(self.N / self.n_clusters))


    def optimal_partition_setting(self):
        """
        :return: Either the optimal n of clusters or the optimal threshold size. In any cases,
        the number returned by this function is the same, because it finds the value at which
        they are (expected to be) equal.
        """
        return int(round(minimize(
            lambda x: max(x, self.N / x), 1, method='nelder-mead', options={ 'xtol': 1e-8, 'disp': True }).x[0]))


    def get_init_info(self):
        init_info = super(SketchRefine, self).get_init_info()
        sketch_refine_init_info = {
            "clust_setting": self.clust_setting,
            "epsilon": self.epsilon if self.epsilon is not None else -1,
            "clustering_online_wc_time": self.clustering_online_wc_time,
            "clustering_online_ct_time": self.clustering_online_ct_time,
            "dbms_index_wc_time": self.dbms_index_wc_time,
            "dbms_index_ct_time": self.dbms_index_ct_time,
            "repr_wallclock_time": self.repr_wallclock_time,
            "repr_cputicks_time": self.repr_cputicks_time,
            "clustering_recover_w_index_wc_time": self.clustering_recover_w_index_wc_time,
            "clustering_recover_w_index_ct_time": self.clustering_recover_w_index_ct_time,
            "indexing_wc_time": self.indexing_wc_time,
            "indexing_ct_time": self.indexing_ct_time,
            "recover_clustering_online_wc_time": self.recover_clustering_online_wc_time,
            "recover_clustering_online_ct_time": self.recover_clustering_online_ct_time,
            "n_clusters": self.n_clusters,
            "clust_attrs": self.clust_attrs,
            "clust_size_threshold": self.clust_size_threshold,
            "n_tuples_per_cid": self.n_tuples_per_cid,
            "n_repres_per_cid": self.n_repres_per_cid,
            "max_radius": self.max_radius,
            "max_satisfied_epsilons": self.max_satisfied_epsilons,
            "n_violating_cids": self.n_violating_cids,
            "worst_violated_epsilons": self.worst_violated_epsilons,
            "avg_satisfied_epsilons": self.avg_satisfied_epsilons,
            "avg_all_epsilons": self.avg_all_epsilons,
        }
        for key in sketch_refine_init_info:
            setattr(init_info, key, sketch_refine_init_info[key])
        return init_info


    def calculate_partitioning_quality(self):
        print "Calculating clustering quality..."

        if self.query.objective is None:
            print "Query has no objective, cluster quality makes no sense."
            return

        max_overal_radius = 0.0
        empirical_epsilons = []
        sql = "SELECT * FROM {SR}.{repr_table}".format(SR=self.sr_schema, repr_table=self.repr_table_name)
        for r in self.db.sql_query(sql):
            max_overal_radius = max(max_overal_radius, r.radius)
            # The empirical epsilon of a cluster is the maximum (worst) empirical epsilon on each attribute
            empirical_epsilon = max(
                getattr(r, "emp_eps_{obj}_{attr}".format(
                    attr=attr,
                    obj="max" if isinstance(self.query.objective.sense, ObjectiveSenseMAX)
                    else "min" if isinstance(self.query.objective.sense, ObjectiveSenseMIN)
                    else None))
                for attr in self.clust_attrs)
            if empirical_epsilon is not None:
                empirical_epsilons.append(empirical_epsilon)

        if isinstance(self.query.objective.sense, ObjectiveSenseMAX):
            # 0 <= epsilon < 1
            def eps_valid(_e):
                return 0 <= _e < 1
        elif isinstance(self.query.objective.sense, ObjectiveSenseMIN):
            # epsilon >= 0
            def eps_valid(_e):
                return 0 <= _e
        else:
            raise Exception("Unknown objective sense.")

        try:
            max_satisfied_epsilons = max(e for e in empirical_epsilons if eps_valid(e))
        except ValueError:
            max_satisfied_epsilons = None
        try:
            worst_violated_epsilons = max(e for e in empirical_epsilons if not eps_valid(e))
        except ValueError:
            worst_violated_epsilons = None
        n_violating_cids = sum(1 for e in empirical_epsilons if not eps_valid(e))
        if max_satisfied_epsilons is not None:
            avg_satisfied_epsilons = sum(e for e in empirical_epsilons if eps_valid(e)) / self.n_clusters
        else:
            avg_satisfied_epsilons = None

        avg_all_epsilons = sum(empirical_epsilons) / self.n_clusters

        print "MAX OVERAL RADIUS =", max_overal_radius
        print "MAX SATISFIED EPSILON =", max_satisfied_epsilons
        print "NUMBER OF VIOLATING CLUSTERS:", n_violating_cids
        print "PERCENTAGE OF VIOLATING CLUSTRS:", n_violating_cids / self.n_clusters * 100
        print "WORST VIOLATED EPSILON =", worst_violated_epsilons
        print "AVG SATISFIED EPSILON =", avg_satisfied_epsilons
        print "AVG ALL EPSILON =", avg_all_epsilons

        self.max_radius = max_overal_radius
        self.max_satisfied_epsilons = max_satisfied_epsilons
        self.n_violating_cids = n_violating_cids
        self.worst_violated_epsilons = worst_violated_epsilons
        self.avg_satisfied_epsilons = avg_satisfied_epsilons
        self.avg_all_epsilons = avg_all_epsilons


    @property
    def cids(self):
        assert self.n_tuples_per_cid is not None
        assert self.n_repres_per_cid is not None
        assert set(self.n_tuples_per_cid.keys())==set(self.n_repres_per_cid.keys()),\
            "Cluster ids differ:\n{}\n{}".format(
                sorted(self.n_tuples_per_cid.keys()),
                sorted(self.n_repres_per_cid.keys()))
        return sorted(self.n_tuples_per_cid.iterkeys())


    def get_one_feasible_package_hook(self, *args, **kwargs):
        raise NotImplemented


    def enumerate_feasible_packages_hook(self, *args, **kwargs):
        raise NotImplemented


    def enumerate_package_space_hook(self, *args, **kwargs):
        raise NotImplemented


    def get_optimal_package_hook(self, strategy, *args, **kwargs):
        if strategy=="greedy-backtrack":
            return self.greedy_backtrack_search(*args, **kwargs)

        else:
            raise Exception("Unsupported strategy: '{}'".format(strategy))


    def init_partitioning(self, reset):
        log("Initing clustering...")

        # Drop pg index to avoid slowing down the sketch_refine process
        self.drop_partitioning_dbms_index()

        ###########################################################################
        # IMPORTANT NOTE:
        # When you add more column to the dataset table, remember to update them
        # in load_queries_from_file to avoid putting these to the data attributes.
        ###########################################################################

        # Add cluster ID column (if does not exists)
        clust_col_exists = sql_table_column_exists(self.db, self.schema_name, self.table_name, "cid", CID_TYPE.lower())
        if not clust_col_exists:
            # Create sketch_refine column and initialize it
            if not sql_table_column_exists(self.db, self.schema_name, self.table_name, "cid"):
                log("Adding column cid...")
                self.db.sql_update(
                    "ALTER TABLE {S}.{D} ADD COLUMN cid {cid_type} DEFAULT {default_cid}::{cid_type_cast}".format(
                        S=self.schema_name,
                        D=self.table_name,
                        cid_type=self.cid_type,
                        cid_type_cast=self.cid_type_cast,
                        default_cid=self.default_cid,
                    ))

            else:
                log("Setting correct cid data type...")
                self.db.sql_update(
                    "ALTER TABLE {S}.{D} DROP COLUMN cid CASCADE; "
                    "ALTER TABLE {S}.{D} ADD COLUMN cid {cid_type} DEFAULT {default_cid}::{cid_type_cast}".format(
                        S=self.schema_name,
                        D=self.table_name,
                        cid_type=self.cid_type,
                        cid_type_cast=self.cid_type_cast,
                        default_cid=self.default_cid,
                    ))

        if reset and clust_col_exists:
            log("Resetting clustering colums: cid...")
            self.db.sql_update(
                "ALTER TABLE {S}.{D} DROP COLUMN cid CASCADE; "
                "ALTER TABLE {S}.{D} ADD COLUMN cid {cid_type} DEFAULT {default_cid}::{cid_type_cast}".format(
                    S=self.schema_name,
                    D=self.table_name,
                    cid_type=self.cid_type,
                    cid_type_cast=self.cid_type_cast,
                    default_cid=self.default_cid,
                ))

        log("Init partitioning: done.")


    def perform_online_partitioning(self):
        print
        print "*" * 50
        print " ONLINE PARTITIONING STARTED"
        print "*" * 50

        ###########################################
        # Initialize partitioning
        ###########################################
        # Check if "marginal clustering" applies
        marginal_clustering_applies = False

        current_clustering = list(self.db.sql_query(
            "SELECT * FROM {SR}.currently_loaded_clustering "
            "WHERE table_name = %s".format(SR=self.sr_schema),
            self.schema_name + "." + self.table_name))

        if len(current_clustering) > 0 and self.mode_marginal:
            if self.clust_size_threshold is not None and\
                            current_clustering[0].effective_max_clust_size is not None and\
                            current_clustering[0].clust_algo==self.clust_algo and\
                            current_clustering[0].clust_attrs==self.clust_attrs and\
                            current_clustering[0].epsilon==self.epsilon and\
                            current_clustering[0].effective_max_clust_size >= self.clust_size_threshold:
                # In this case we don't need to cluster from scratch, we can continue from current clustering
                # need_to_reset_clustering = False
                marginal_clustering_applies = True
                log("Marginal clustering applies.")
            elif self.min_n_clusts is not None and\
                            current_clustering[0].effective_n_clusts is not None and\
                            current_clustering[0].clust_algo==self.clust_algo and\
                            current_clustering[0].clust_attrs==self.clust_attrs and\
                            current_clustering[0].epsilon==self.epsilon and\
                            current_clustering[0].effective_n_clusts <= self.min_n_clusts:
                # In this case we don't need to cluster from scratch, we can continue from current clustering
                # need_to_reset_clustering = False
                marginal_clustering_applies = True
                log("Marginal clustering applies.")

        need_to_reset_clustering =\
            self.last_clust_size_threshold is None or\
            self.last_clust_size_threshold < self.clust_size_threshold or\
            not marginal_clustering_applies
        log("Reset of partitioning is needed." if need_to_reset_clustering else "Not resetting partitioning.")

        # Init partitioning
        self.init_partitioning(reset=need_to_reset_clustering)

        # Perform clustering
        log("Clustering now...")
        self.cluster_data()

        # Store sketch_refine meta-information in memory (n of tuples per cluster)
        self.n_tuples_per_cid = { }

        self.load_stored_partitioning(just_clustered=True)


    def prep_completed_partitioning(self, store):
        """
        Create the DBMS indexes, choose representatives, and remember current dataset loaded sketch_refine
        """
        if store:
            self.store_partitioning()
        self.create_partitioning_dbms_index()
        self.load_partitioning_metainfo()
        self.last_clust_size_threshold = self.clust_size_threshold

        self.db.sql_update(
            "WITH upsert AS ("
            "   UPDATE {SR}.currently_loaded_clustering SET "
            "       clust_table_name = %(clust_table_name)s, "
            "       clust_setting = %(clust_setting)s, "
            "       clust_algo = %(clust_algo)s, "
            "       clust_attrs = %(clust_attrs)s,"
            "       effective_max_clust_size = %(effective_max_clust_size)s, "
            "       effective_n_clusts = %(effective_n_clusts)s "
            "   WHERE "
            "       table_name = %(table_name)s "
            "   RETURNING *)"
            "INSERT INTO {SR}.currently_loaded_clustering ("
            "   table_name, "
            "   clust_table_name, "
            "   clust_setting, "
            "   epsilon, "
            "   clust_algo, "
            "   clust_attrs, "
            "   effective_max_clust_size, "
            "   effective_n_clusts) "
            "SELECT "
            "   %(table_name)s, "
            "   %(clust_table_name)s, "
            "   %(clust_setting)s, "
            "   %(epsilon)s, "
            "   %(clust_algo)s, "
            "   %(clust_attrs)s, "
            "   %(effective_max_clust_size)s, "
            "   %(effective_n_clusts)s "
            "WHERE NOT EXISTS (SELECT * FROM upsert)".format(
                SR=self.sr_schema,
            ),
            table_name=self.schema_name + "." + self.table_name,
            clust_table_name=self.stored_clust_table_name,
            clust_setting=self.clust_setting,
            epsilon=self.epsilon,
            clust_algo=self.clust_algo,
            clust_attrs=self.clust_attrs,
            effective_max_clust_size=max(self.n_tuples_per_cid.itervalues()),
            effective_n_clusts=self.n_clusters)

        assert self.n_clusters==len(self.cids)
        print "N of clusters:", self.n_clusters
        if self.clust_algo=="quad":
            print "Exptected n of clusters was:", self.expected_n_partitions()

        self.commit()

        # NOTE: Uncomment if you want to see statistics about the quality of the partitioning (takes time)
        # self.calculate_partitioning_quality()


    def store_partitioning(self):
        if self.use_index:
            store_table = self.index_table_name
        else:
            store_table = self.stored_clust_table_name

        log("Storing clustering{} table '{}'...".format(
            "" if not self.use_index else " index",
            store_table))

        # Create storing table
        self.db.sql_update(
            "DROP TABLE IF EXISTS {SR}.{C}".format(C=store_table, SR=self.sr_schema))
        self.db.sql_update(
            "CREATE TABLE {SR}.{C} ("
            "	id varchar NOT NULL, "
            "	cid {cid_type_cast} NOT NULL DEFAULT {default_cid}::{cid_type_cast} "
            ")".format(
                SR=self.sr_schema,
                C=store_table,
                cid_type=self.cid_type,
                cid_type_cast=self.cid_type_cast,
                default_cid=self.default_cid))

        # Store
        self.db.sql_update(
            "INSERT INTO {SR}.{C} (id, cid) "
            "SELECT D.id, D.cid FROM {S}.{D} D".format(
                SR=self.sr_schema,
                C=store_table,
                S=self.schema_name,
                D=self.table_name))

        # This DBMS index is to speed up loading up this clustering next time
        self.db.sql_update("CREATE INDEX ON {SR}.{C} USING btree (id)".format(
            C=store_table,
            SR=self.sr_schema))

        log("Clustering{} table '{}' stored.".format(
            "" if not self.use_index else " index",
            store_table))


    def create_partitioning_dbms_index(self):
        clust_index_exists = sql_index_exists(self.db, self.schema_name, self.table_name, col_name="cid")

        if clust_index_exists:
            # If index on cid already exists, you don't need to create it
            log("Clustering DBMS index on 'cid' already exists")
            return

        ###########################################
        # Index sketch_refine column (Postgres index)
        ###########################################
        # NOTE: The purpose of this index is to speed up the retrieval of tuples corresponding to each cluster at
        # query time. If you don't have this index, Backtracking (and any other query method) will be significantly
        # slower.
        log("Creating index '{}' on data table for fast cluster tuples retrieval...".format(
            self.clust_dbms_index_name))
        self.dbms_index_wc_time = -time.time()
        self.dbms_index_ct_time = -time.clock()
        self.db.sql_update("CREATE INDEX {index_name} ON {S}.{D} USING btree (cid)".format(
            S=self.schema_name,
            D=self.table_name,
            index_name=self.clust_dbms_index_name))
        self.dbms_index_wc_time += time.time()
        self.dbms_index_ct_time += time.clock()
        log("Index created ({} sec).".format(self.dbms_index_wc_time))


    def drop_partitioning_dbms_index(self):
        log("Dropping DBMS indexes on cid from data table...")

        indexes = sql_get_all_indexes(self.db, self.schema_name, self.table_name)

        found_some = False
        for index_name, col_name, isprimary in indexes:
            if col_name=="cid":  # and not isprimary:
                found_some = True
                log("Dropping DBMS index '{}'...".format(index_name))
                self.db.sql_update("DROP INDEX {schema_name}.{index_name}".format(
                    schema_name=self.schema_name,
                    index_name=index_name))

        if not found_some:
            log("No DBMS indexes found on data table.")


    def commit(self):
        ###########################################
        # Commit
        ###########################################
        log("Committing...")
        pg_commit_wc_time = -time.time()
        pg_commit_ct_time = -time.clock()
        self.db.commit()
        pg_commit_wc_time += time.time()
        pg_commit_ct_time += time.clock()
        log("Commit done ({} sec).".format(pg_commit_wc_time))


    def cluster_data(self):
        if self.clust_algo=="quad":
            clusts = QuadTreePartitioning(
                sr_schema=self.sr_schema,
                db=self.db,
                dataset_size=self.N,
                nbits=self.nbits_clust,
                cid_type_cast=self.cid_type_cast,
                data_table_name=self.schema_name + "." + self.table_name,
                index_table_name=self.index_table_name,
                indexing_attrs=self.indexing_attrs,
                repr_table_name=self.repr_table_name,
                clust_attrs=self.clust_attrs,
                data_attrs=self.dataset_attrs,
                max_clust_size=self.clust_size_threshold,
                min_n_clusters=self.min_n_clusts,
                epsilon=self.epsilon,
                obj_type=str(self.query.objective.sense) if self.query.objective else None)

        else:
            raise Exception("Clustering algorithm name '{}' not supported.".format(self.clust_algo))

        # Perform clustering
        self.clustering_online_wc_time = -time.time()
        self.clustering_online_ct_time = -time.clock()
        clusts.fit()
        self.clustering_online_wc_time += time.time()
        self.clustering_online_ct_time += time.clock()

        # TODO: You can safely remove these variables from this class, and all references to them
        # In the new implementation representatives are directly computed during partitioning,
        # so we don't need to keep track of this time anymore.
        self.repr_wallclock_time = 0
        self.repr_cputicks_time = 0

        # Return labels
        return clusts


    def load_stored_partitioning(self, just_clustered=False):
        """
        Clusterings are stored in the DB. At query time we reload them into the data table.
        The time taken to do this operation is saved, but not very important because it a real application
        the clusterings can simply always remain in their respective data tables.
        """
        # To check whether the dataset is already clustered correctly (so you don't need to load it)
        current_clustering = list(self.db.sql_query(
            "SELECT * FROM {SR}.currently_loaded_clustering "
            "WHERE table_name = %s".format(SR=self.sr_schema), self.schema_name + "." + self.table_name))

        if not self.use_index and just_clustered:
            log("Data was just clustered...")
            self.prep_completed_partitioning(store=True)

        elif self.use_index and just_clustered:
            log("Data was just indexed...")
            # log("After performing indexing, we stop the process. You'll need to call again without setting -r.")
            self.prep_completed_indexing()

        elif not self.use_index and\
                        len(current_clustering) > 0 and\
                        current_clustering[0].clust_table_name==self.stored_clust_table_name:
            log("Data is already clustered correctly, doing nothing...")
            self.prep_completed_partitioning(store=False)

        elif self.use_index and sql_table_exists(self.db, self.sr_schema, self.index_table_name):
            self.init_partitioning(reset=False)

            self.load_clustering_from_index(index_already_in_table=False)

        elif not self.use_index and sql_table_exists(self.db, self.sr_schema, self.stored_clust_table_name):
            self.init_partitioning(reset=False)

            self.recover_clustering_online_wc_time = -time.time()
            self.recover_clustering_online_ct_time = -time.clock()

            # Load up sketch_refine column from sketch_refine table into data table
            log("Loading up stored partitioning into data table...")
            self.db.sql_update(
                "UPDATE {S}.{D} SET cid = C.cid "
                "FROM {SR}.{C} C "
                "WHERE {D}.id = C.id".format(
                    SR=self.sr_schema,
                    S=self.schema_name,
                    D=self.table_name,
                    C=self.stored_clust_table_name))

            self.recover_clustering_online_wc_time += time.time()
            self.recover_clustering_online_ct_time += time.clock()

            log("Stored partitioning loaded up in {} sec.".format(self.recover_clustering_online_wc_time))

            self.prep_completed_partitioning(store=False)

        elif not self.use_index:
            log("Stored partitioning not present. Need to cluster now.")

            # Compute and store the clusters
            log("Partitioning (C={}) with {} on attributes: {}...".format(
                self.n_clusters, self.clust_algo, self.clust_attrs))

            self.perform_online_partitioning()

        else:
            raise Exception("Inconsistent case.")

        # Sanity check
        n_tuples = sum(self.n_tuples_per_cid.itervalues())
        assert n_tuples==self.N


    def load_partitioning_metainfo(self, just_created=False):
        """
        Load information about the partitioning (number of tuples and representatives per partition)
        """
        # For now only support average point
        assert len(self.use_representatives)==1 and self.use_representatives[0].__name__=="avgp"

        if sql_table_exists(self.db, self.sr_schema, self.repr_table_name):
            log("Loading meta info about partitions...")

            # Load up meta-info about representatives
            reprs = self.db.sql_query(
                "SELECT cid, cid_size, COUNT(*) AS n_repr "
                "FROM {SR}.{repr_table} "
                "GROUP BY cid, cid_size".format(
                    SR=self.sr_schema,
                    repr_table=self.repr_table_name))

            self.n_repres_per_cid = { }
            self.n_tuples_per_cid = { }
            for r in reprs:
                self.n_repres_per_cid[r.cid] = getattr(r, "n_repr")
                self.n_tuples_per_cid[r.cid] = getattr(r, "cid_size")

            self.n_clusters = len(self.n_repres_per_cid)

            assert self.n_clusters==len(self.n_tuples_per_cid)
            assert sum(self.n_tuples_per_cid.itervalues())==self.N,\
                "N = {}, but sum(self.n_tuples_per_cid.itervalues()) = {}".format(
                    self.N, sum(self.n_tuples_per_cid.itervalues()))

        elif just_created:
            raise Exception("Representative table '{}' should exist.".format(self.repr_table_name))

        else:
            assert self.already_partitioned, "This can only happen when the dataset was pre-clustered."
            log("Dataset was pre-clustered but representative table is not present yet. Creating now...")

            clusts = QuadTreePartitioning(
                sr_schema=self.sr_schema,
                db=self.db,
                dataset_size=self.N,
                nbits=self.nbits_clust,
                cid_type_cast=self.cid_type_cast,
                data_table_name=self.schema_name + "." + self.table_name,
                index_table_name=self.index_table_name,
                indexing_attrs=self.indexing_attrs,
                repr_table_name=self.repr_table_name,
                clust_attrs=self.clust_attrs,
                data_attrs=self.dataset_attrs,
                max_clust_size=self.clust_size_threshold,
                min_n_clusters=self.min_n_clusts,
                epsilon=self.epsilon,
                obj_type=str(self.query.objective.sense))

            clusts.fit(only_representatives=True)
            self.load_partitioning_metainfo(just_created=True)
            log("Represetantive table created.")


    def greedy_backtrack_search(self, start_empty):
        from src.paql_eval.sketch_refine.greedy_backtracking import GreedyBacktracking, GreedyBacktrackRunInfo

        self.current_run_info.strategy_run_info = GreedyBacktrackRunInfo()

        gbt_search = GreedyBacktracking(self)
        final_partial_package = gbt_search.run(start_empty)

        if final_partial_package is not None:
            # Solution must be complete (all clusters solved)
            assert isinstance(final_partial_package, PartialPackage)
            iscomplete = final_partial_package.is_complete
            assert iscomplete, "PartialSolution produced by full backtrack is not a full candidate solution."

            print "Converting to Package..."
            tt = -time.time()
            self.current_run_info.timelimit = None
            package = final_partial_package.to_candidate_package()
            tt += time.time()
            print "done ({} s).".format(tt)

            return package
        else:
            raise ReducedSpaceInfeasible(message="Full Backtracking Search failed")
