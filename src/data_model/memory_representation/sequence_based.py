from __future__ import division

import numpy as np

from src.data_model.memory_representation.__package import Package
from src.paql.constraints import CGlobalConstraint
from src.utils.log import *
from src.utils.sorted_collection import SortedCollection


class SequenceBasedPackage(Package):
    """
    A sequence-based package represents a package in main memory as a *sorted* collection of "sequence IDs".

    A sequence ID is a unique identifier of a tuple from the input table. It is not any of the existing
    attributes of the table (more importantly, not the primary key). It is not a "oid" either. The sequence
    ID is constructed by sorting the table by the primary key (id) and then assigning a sequential number,
    starting from 1, to each tuple in that order.

    This class only stores the sequence IDs of the tuples in the package.
    """

    def __init__(self, search, combination=None):
        """
        :param search: The Search object which is using this candidate pacakge.
        """
        super(SequenceBasedPackage, self).__init__(search)

        self._combination = None
        self._is_feasible = None
        self._objective_value = None

        if combination is not None:
            self.combination = combination


    def invalidate(self):
        self._combination = None
        self._is_feasible = None
        self._objective_value = None
        self.search = None
        self.table_name = None


    @staticmethod
    def from_ids(search, ids):
        """
        Returns a new CandidatePackage where the combination is given by ids and not by seqs (as in __init__)
        """
        combo = search.package_table.get_seqs_by_ids(ids)
        return SequenceBasedPackage(search, combo)


    @property
    def combination(self):
        return list(self._combination)


    @combination.setter
    def combination(self, combination):
        self._combination = SortedCollection(iterable=combination)
        assert all(type(c) is int or type(c) is long for c in self.combination), [type(c) for c in self.combination]
        self._is_feasible = None
        self._objective_value = None


    def __contains__(self, item):
        return item in self.combination


    def __hash__(self):
        return hash(tuple(self.combination))


    def __eq__(self, other):
        if other is None:
            return False
        return self.combination == other.combination


    def __ne__(self, other):
        if other is None:
            return True
        return self.combination != other.combination


    def __lt__(self, other):
        obj_type = str(self.search.query.objective.sense)
        if obj_type == "maximize":
            return self.get_objective_value() < other.get_objective_value()
        elif obj_type == "minimize":
            return self.get_objective_value() > other.get_objective_value()
        else:
            raise Exception("Objective type '{}' not recognized!".format(obj_type))


    def __gt__(self, other):
        """
        Returns whether this candidate is *strictly* better than 'other' candidate for the objective value.
        """
        assert self.search.query.objective is not None
        obj_type = str(self.search.query.objective.sense)
        if obj_type == "maximize":
            return self.get_objective_value() > other.get_objective_value()
        elif obj_type == "minimize":
            return self.get_objective_value() < other.get_objective_value()
        else:
            raise Exception("Objective type '{}' not recognized!".format(obj_type))


    def __le__(self, other):
        if other is None:
            return False
        return self.combination == other.combination or self.__lt__(other)


    def __ge__(self, other):
        if other is None:
            return True
        return self.combination == other.combination or self.__gt__(other)


    def __len__(self):
        return len(self.combination)


    def __repr__(self):
        # FIXME: Check if returning a str() here is legal.
        return str(tuple(sorted(self.combination)))


    def materialize(self, attrs=None):
        self.search.package_table.materialize(self, attrs)


    def drop(self, *args, **kwargs):
        self.search.package_table.drop()


    def delete(self, *args, **kwargs):
        self.search.package_table.delete()


    def iter_tuples(self, attrs=None):
        log("materializing...")
        self.materialize(attrs)
        log("materializing: done.")
        return self.search.db.sql_query("SELECT * FROM {} ORDER BY id".format(self.table_name))


    def get_objective_value(self):
        """
        Computes the objective function for the current candidate package.
        """
        assert self.search.query.objective is not None

        if self._objective_value is not None:
            assert type(self._objective_value) is float
            return self._objective_value

        self.materialize()
        sql = "SELECT {} FROM {}".format(
            self.search.query.objective.get_aggregate().get_sql(),
            self.table_name)
        result = self.search.db.sql_query(sql).next()[0]
        self._objective_value = float(result)
        return self._objective_value


    def update_global_scores(self):
        self.materialize()
        self.gscores = {} # TODO: Move this in init or in class init space
        # TODO: The idea here is to assign a score to a candidate based on how well it satisfies each constraint.

        gscores = self.get_coalesced_global_scores()

        for i, ((aggr, attr), (lb, ub)) in enumerate(self.search.query.coalesced_gcs.iteritems()):
            score = gscores[i]
            self.gscores[(aggr, attr)] = lb, ub, score


    def get_coalesced_global_scores(self):
        """
        Computes the global queries for the current candidate package with respect to the current query.
        """
        return self.get_coalesced_global_scores_wrt_query(self.search.query)


    def get_uncoalesced_global_scores(self):
        """
        Computes the global queries for the current candidate package with respect to the current query.
        """
        return self.get_uncoalesced_global_scores_wrt_query(self.search.query)


    def get_coalesced_base_scores(self):
        return self.get_coalesced_base_scores_wrt_query(self.search.query)


    def get_uncoalesced_base_scores(self):
        return self.get_uncoalesced_base_scores_wrt_query(self.search.query)


    def get_coalesced_global_scores_wrt_query(self, query):
        """
        Computes the global queries for the current candidate package with respect to the specified query.
        """
        return self.compute_global_scores(query.coalesced_gcs)


    def get_uncoalesced_global_scores_wrt_query(self, query):
        """
        Computes the global queries for the current candidate package with respect to the specified query.
        """
        return self.compute_global_scores((aggr, attr) for (aggr, attr), op, n in query.uncoalesced_gcs)


    def get_coalesced_base_scores_wrt_query(self, query):
        attrs = [ attr for attr in query.coalesced_bcs.iterkeys() ]
        for scores in self.get_base_scores_for(attrs):
            yield scores


    def get_uncoalesced_base_scores_wrt_query(self, query):
        attrs = [ attr for attr, op, n in query.uncoalesced_bcs ]
        for scores in self.get_base_scores_for(attrs):
            yield scores


    def compute_global_scores(self, c_gcs):
        simple_cgcs = []
        groupbyall_cgcs = []

        print "COALESCED GLOBAL CONSTRAINTS:"

        for cgc in c_gcs:
            assert isinstance(cgc, CGlobalConstraint)

            print ">>>>", cgc

            if cgc.is_simple_linear():
                simple_cgcs.append(cgc)

            else:
                groupbyall_cgcs.append(cgc)

        self.materialize()
        aggr_simple_scores = self._compute_global_scores_simple_sql(simple_cgcs)
        aggr_groupby_all_scores = self._compute_global_scores_groupbyall_sql(groupbyall_cgcs)
        return aggr_simple_scores + aggr_groupby_all_scores


    def _compute_global_scores_simple_sql(self, simple_cgcs):
        """
        Returns a single tuple containing all global package values for all (coalesced) global constraints.
        """
        sqls = [
            str(cgc.expr) + " AS " + str(cgc.expr).replace("(", "_").replace(")", "").replace("*", "")
            for cgc in simple_cgcs
        ]

        if len(simple_cgcs) == 0:
            return []

        scores_sql = "SELECT {aggregates} FROM {R}".format(
            aggregates=",".join(sqls),
            R=self.table_name)

        scores = self.search.db.sql_query(scores_sql).next()

        return zip(simple_cgcs, scores)


    def _compute_global_scores_groupbyall_sql(self, groupbyall_cgcs):
        """
        Returns a single tuple containing all global package values for all (coalesced) global constraints.
        """
        scores = []
        for cgc in groupbyall_cgcs:
            cgcstr = str(cgc.expr)
            gr = cgcstr.upper().find("GROUP BY")
            aggr_sql = cgcstr[:gr] + "AS " + \
                       cgcstr[:gr].replace("(", "").replace(")", "")\
                           .replace("*", "").replace("/", "")\
                           .replace("+", "").replace("-", "").replace(".", "").replace("0", "")
            gby_sql = cgcstr[gr:]
            score_sql = ("SELECT " + aggr_sql + "FROM {R} " + gby_sql).format(R=self.table_name)
            cgc_scores = [ s[0] for s in self.search.db.sql_query(score_sql) ]
            for score in cgc_scores:
                scores.append((cgc, score))

        return scores


    def get_base_scores_for(self, attrs):
        if len(attrs):
            self.materialize()
            scores_sql = "SELECT {} from {}".format(
                ",".join(attrs),
                self.table_name)
            for scores in self.search.db.sql_query(scores_sql):
                yield list(scores)


    @staticmethod
    def generate_empty_candidate(search):
        return SequenceBasedPackage(search, tuple())


    @staticmethod
    def generate_full_candidate(search):
        return SequenceBasedPackage(search, range(1, search.N + 1))


    @staticmethod
    def generate_random_candidate(search):
        """
        Generates random candidate in the plausibly valid paql_eval space (considering pruning if it is on).
        """
        def include_random_tuples():
            for x in range(1, search.N + 1):
                assert len(combination) <= search.ub
                if len(combination) == search.ub:
                    break

                # If r < p, you add the tuple to the current package.
                r = np.random.random()
                if r < p and x not in combination:
                    # Add tuple to new combo
                    combination.append(x)

        p = np.random.random()

        # For each tuple in the input table, you generate a random number r in (0,1).
        combination = []

        # Keep adding random tuples to reach the cardinality lower bound
        while len(combination) < search.lb:
            include_random_tuples()

        # Only if you haven't reached the cardinality upper bound, try to include more random tuples
        if len(combination) < search.ub:
            include_random_tuples()

        return SequenceBasedPackage(search, combination)


    def gen_missing_tuple_seqs(self):
        """
        Yields one missing tuple (seq number) at a time
        """
        if len(self.combination) < self.search.ub:
            for missing_tuple in xrange(1, self.search.N + 1):
                if missing_tuple not in self.combination:
                    yield missing_tuple


    def is_valid(self):
        """
        Return whether this candidate package is feasible, i.e., if it satisfies all the constraints (base and global).
        """
        # Check if already computed
        if self._is_feasible is not None:
            assert type(self._is_feasible) == bool
            return self._is_feasible

        # Check all global constraints
        global_scores = self.get_coalesced_global_scores()

        for cgc, score in global_scores:
            assert isinstance(cgc, CGlobalConstraint)
            print "Checking Constraint", cgc, "SCORE:", score
            if not cgc.lb <= score <= cgc.ub:
                print "! INVALID !"
                self._is_feasible = False
                return self._is_feasible
            else:
                "VALID"

        self._is_feasible = True
        return self._is_feasible


    def is_cardinality_valid(self):
        return self.search.lb <= len(self.combination) <= self.search.ub


    def is_be_wrt_error(self, other, error_func, paql_query):
        objective_type = self.search.query.objective['type']

        self_error = error_func(self)(paql_query) + 1
        other_error = error_func(other)(paql_query) + 1

        if objective_type=='maximize':
            return self.get_objective_value() / self_error >= \
                   other.get_objective_value() / other_error
        elif objective_type=='minimize':
            return self.get_objective_value() * self_error <= \
                   other.get_objective_value() * other_error
        else:
            raise Exception('Objective type not recognized.')
