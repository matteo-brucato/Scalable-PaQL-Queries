# coding=utf-8
from __future__ import division

from overrides.overrides import overrides

from src.dbms.db import DBConnection
from src.dbms.db import DBCursor
from src.dbms.dbms_settings import data_dbms_settings
from src.experiments.paql_eval.main import PaQLEvalExperiment
from src.paql.package_query import PackageQuery
from src.paql_eval.search_process import *
from src.paql_eval.sketch_refine.sketch_refine import *
from src.utils.utils import avgp, iter_subsets, iter_supersets




class SketchRefineExperiment(PaQLEvalExperiment):
    @property
    @overrides
    def description(self):
        return "SketchRefine Algorithm experiment."

    @overrides
    def set_args(self):
        super(SketchRefineExperiment, self).set_args()

        self.arg_parser.add_argument(
            "-c", "--partitioning",
            default="quad",
            help="The partitioning algorithm to be used. Options are: {}".format(
                ", ".format("'{}'".format(clust_algo_name)
                            for clust_algo_name in SketchRefine.clust_algo_names)))
        self.arg_parser.add_argument(
            "-C", "--partitioning-setting", required=False,
            help="Either: (1) The number of clusters to construct; "
                 "or (2) The threshold size of each cluster. "
                 "Each sketch_refine algorithm will use one of the two. "
                 "Set it to `*' to let it be decided automatically at runtime "
                 "based on each query. "
                 "You can separate multiple sketch_refine settings with `,'.")
        self.arg_parser.add_argument(
            "-a", "--attributes", dest="clust_attrs_sets",
            default="!",
            help="Which attributes to cluster on. "
                 "Separate single attributes with `,', and sets of attributes with `;'. "
                 "Use * to tell the runtime to use all attributes of the query, "
                 "or ! to use all the attributes of the input table.")
        self.arg_parser.add_argument(
            "--eps", type=float, default=-1,
            help="Epsilon value for quality guarantee. Don't set or set to -1 if you don't want the guarantee.")
        self.arg_parser.add_argument(
            "-t", "--vartype", default="int",
            help="The type of lp variables used to solve for representative tuples.")

        # Booleans
        self.arg_parser.add_argument(
            "--already-partitioned", action="store_true",
            help="Set this if you know for sure that the dataset is already properly partitioned for all the queries. "
                 "This will skip all the checks.")
        self.arg_parser.add_argument(
            "-r", "--repartition", action="store_true",
            help="Set if you want to repartition all input tables for all queries.")
        self.arg_parser.add_argument(
            "-f", "--store-all-lp-problems", action="store_true",
            help="Whether to store all lp problems in files. "
                 "Files are stored in the run directory.")
        self.arg_parser.add_argument(
            "--start-empty", action="store_true",
            help="Set this option if you do not want to firstly solve for the representatives alone but, "
                 "instead, you want to solve the first cluster together with the representatives. "
                 "Default is false, and it will firstly try to solve the representatieves alone.")
        self.arg_parser.add_argument(
            "--nomarginal", action="store_true")

        # Defaults
        self.arg_parser.set_defaults(already_partitioned=False)
        self.arg_parser.set_defaults(repartition=False)
        self.arg_parser.set_defaults(store_all_lp_problems=False)
        self.arg_parser.set_defaults(start_empty=False)
        self.arg_parser.set_defaults(nomarginal=False)


    @overrides
    def run(self, *args):
        # Create run directory if we need to store all LP problems to files
        if self.args.store_all_lp_problems:
            self.init_data_folder()

        # Set database where the input data reside
        self.set_datadb(DBCursor(DBConnection(**data_dbms_settings)))

        print "=" * 70
        print "| NEW RUN {:<58}|".format("")
        print "=" * 70

        ############################################################################################################
        # Evaluate each query
        ############################################################################################################
        q = PackageQuery.from_paql(open(self.args.query_file).read())
        self.sketch_refine(q)


    def sketch_refine(self, q):
        # The list of representatives to use
        use_representatives = [avgp]  # Use only the average point (centroid)

        query_attrs = q.get_attributes()
        print "query attrs:", query_attrs
        data_attrs = sorted(set(q.get_data_attributes(self.datadb)) - {"cid", "cid_size", "radius"})
        assert data_attrs, "data attrs: {}".format(data_attrs)

        # Partitioning attributes
        if self.args.clust_attrs_sets.startswith("all-query-subsets"):
            if self.args.clust_attrs_sets.startswith("all-query-subsets-td"):
                order = "top-down"
            elif self.args.clust_attrs_sets.startswith("all-query-subsets-bu"):
                order = "bottom-up"
            else:
                raise Exception
            n_variations = int(self.args.clust_attrs_sets.split("-")[-1])
            # Run all *subsets* of the query's relevant attributes
            partitioning_attribute_sets = [
                sorted(partitioning_attrs)
                for partitioning_attrs in iter_subsets(query_attrs, order, n_variations)
            ]

        elif self.args.clust_attrs_sets.startswith("all-query-supersets"):
            if self.args.clust_attrs_sets.startswith("all-query-supersets-td"):
                order = "top-down"
            elif self.args.clust_attrs_sets.startswith("all-query-supersets-bu"):
                order = "bottom-up"
            else:
                raise Exception
            n_variations = int(self.args.clust_attrs_sets.split("-")[-1])
            # Run all *supersets* of the query's relevant attributes
            partitioning_attribute_sets = [
                sorted(partitioning_attrs)
                for partitioning_attrs in iter_supersets(query_attrs, data_attrs, order, n_variations)
            ]

        elif self.args.clust_attrs_sets.startswith("all-subsets"):
            if self.args.clust_attrs_sets.startswith("all-subsets-td"):
                order = "top-down"
            elif self.args.clust_attrs_sets.startswith("all-subsets-bu"):
                order = "bottom-up"
            else:
                raise Exception
            n_variations = int(self.args.clust_attrs_sets.split("-")[-1])
            # Run all *subsets* of the dataset attributes
            partitioning_attribute_sets = [
                sorted(partitioning_attrs)
                for partitioning_attrs in iter_subsets(data_attrs, order, n_variations)
            ]

        elif self.args.clust_attrs_sets.strip()=="*":
            # Run only one set of attributes, consisting of all the query's relevant attributes
            partitioning_attribute_sets = [sorted(query_attrs)]

        elif self.args.clust_attrs_sets.strip()=="!":
            # Run only one set of attributes, consisting of all the data attributes
            partitioning_attribute_sets = [sorted(data_attrs)]

        else:
            # Use the specified query attributes
            partitioning_attribute_sets = [
                sorted(set(attr.strip() for attr in partitioning_attrs.split(",")))
                for partitioning_attrs in self.args.clust_attrs_sets.split(":")
            ]

        assert len(partitioning_attribute_sets) > 0, "No setting for partitioning attributes."

        # Partitioning quality requirement
        eps = self.args.eps

        # Partitioning settings
        partitioning_settings = [nc for nc in self.args.partitioning_setting.split(",")]

        # Evaluate query for each of the settings and attribute sets
        for partitioning_attrs in partitioning_attribute_sets:
            for partitioning_setting in partitioning_settings:
                self.perform_sketch_refine(
                    q, partitioning_setting, partitioning_attrs, data_attrs, query_attrs,
                    use_representatives, eps)

        print "=" * 70
        print "| RUN FINISHED {:<50}|".format("", "")
        print "=" * 70


    def perform_sketch_refine(self, q, clust_setting, clust_attrs, dataset_attrs, query_attrs, repres, eps):
        print "/-----------------{:-<55}\\".format("")
        print "| Clust algo:     {:<55}|".format(self.args.partitioning)
        print "| Clust setting:  {:<55}|".format(clust_setting)
        print "| Clust attrs:    {:<55}|".format(",".join(clust_attrs))
        print "| Epsilon:        {:<55}|".format(eps)
        print "\-----------------{:-<55}/".format("")

        assert set(clust_attrs).issubset(set(dataset_attrs)),\
            "Clustering attribute set ({}) should be subset of the dataset attributes ({}).".format(
                clust_attrs, dataset_attrs)

        # SketchRefine Algorithm with Backtracking
        __init__kwargs = {
            "use_connection": self.datadb.connection,
        }
        init_kwargs = {
            "query": q,
            "clust_attrs": clust_attrs,
            "query_attrs": query_attrs,
            "dataset_attrs": dataset_attrs,
            "indexing_attrs": [],
            "clust_algo": self.args.partitioning,
            "clust_setting": clust_setting,
            "epsilon": eps,
            "repartition": self.args.repartition,
            "vartype": self.args.vartype,
            "use_representatives": repres,
            "use_index": False,
            "store_lp_problems_dir": self._test_results_dir,
            "mode_marginal": not self.args.nomarginal,
            "already_partitioned": self.args.already_partitioned }
        search_kwargs = {
            "strategy": "greedy-backtrack",
            "start_empty": self.args.start_empty,
            "timelimit": self.args.time_limit }
        results =\
            get_optimal_package_in_subprocess_and_monitor_memory_usage(
                SketchRefine, __init__kwargs, init_kwargs, search_kwargs, self.args.init_only, self.args.mem_limit)

        self.end(results)




tests = [
    SketchRefineExperiment,
]
