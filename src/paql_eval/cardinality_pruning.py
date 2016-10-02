from logging import debug
import math
import time
import sys

from src.utils.utils import n_of_subsets
from ..dbms.db import DBConnection


def create_sorted_table(db, table_name, attribute_name):
    # This is a sort of materialized view, very fastly computed by postgres
    # It is temporary, as it will be destroyed at the end of the session
    print "Create (temporary) sorted table"
    start = time.time()

    sql = "DROP TABLE IF EXISTS {0}_sorted;".format(table_name)
    db.sql_update(sql)
    sql = """CREATE TEMP TABLE {0}_sorted (id,{1}) AS
    SELECT row_number() OVER(ORDER BY {1} ASC),{1} FROM {0} ORDER BY {1};""".format(table_name, attribute_name)
    db.sql_update(sql)
    ### SLIGHT IMPROVEMENT
    sql = """CREATE INDEX ON {0}_sorted USING btree({1});""".format(table_name, attribute_name)
    db.sql_update(sql)

    print time.time() - start, "seconds to create (temp) sorted table."


def compute_prefix_sum(db, table_name, attribute_name):
    sql = """
    SELECT {1}, (SELECT sum(T2.{1}) FROM {0}_sorted T2 WHERE T2.id <= T1.id) AS partial_prefix_sum
    FROM {0}_sorted T1 ORDER BY partial_prefix_sum ASC;""".format(table_name, attribute_name)
    print sql
    return db.sql_query(sql)


def compute_inverse_prefix_sum(db, table_name, attribute_name):
    sql = """
    SELECT {1}, (SELECT sum(T2.{1}) FROM {0}_sorted T2 WHERE T2.id >= T1.id) AS partial_prefix_sum
    FROM {0}_sorted T1 ORDER BY partial_prefix_sum ASC;""".format(table_name, attribute_name)
    #print sql
    return db.sql_query(sql)


def compute_min_max(db, table_name, attribute_name):
    sql = "SELECT min({1}), max({1}) FROM {0}".format(table_name, attribute_name)
    #print sql
    return db.sql_query(sql).next()


def get_from_sorted_table(db, table_name, attribute_name, i):
    sql = "SELECT {1} FROM {0}_sorted WHERE id={2}".format(table_name, attribute_name, i)
    return db.sql_query(sql)[0][0]




####### SUM CARDINALITY CONSTRAINTS #########
def run_minmax_based_sum_pruning(db, table_name, attribute_name, N, a, b):
    minX, maxX = compute_min_max(db, table_name, attribute_name)
    print table_name, attribute_name, minX, maxX

    assert minX >= 0 and maxX >= 0, "Cardinaity pruning is not yet working with negative data."

    # FIXME: What happens if minX=0 or maxX=0?? Is this still correct?
    if maxX:
        n1 = int(max(math.ceil(a/float(maxX)), 0))
    else:
        n1 = 0

    if minX:
        n4 = int(min(math.floor(b/float(minX)), N))
    else:
        n4 = N

    if n1 > n4:
        n1 = n4

    return n1, n4


def run_prefixsum_based_sum_pruning(db, table_name, attribute_name, N, a, b):
    create_sorted_table(db, table_name, attribute_name)

    prefix_sum = compute_prefix_sum(db, table_name, attribute_name)
    inverse_prefix_sum = compute_inverse_prefix_sum(db, table_name, attribute_name)

    print prefix_sum
    print inverse_prefix_sum

    n1_not_found = n4_not_found = True
    print prefix_sum
    print inverse_prefix_sum
    print a, b
    n1 = N+1 #### TODO: FIX THIS!
    n4 = N+1 #### TODO: FIX THIS!
    for n in range(N):
        #print inverse_prefix_sum[n][1]
        if inverse_prefix_sum[n][1] >= a and n1_not_found:
            n1 = n+1
            n1_not_found = False
        if prefix_sum[n][1] > b and n4_not_found:
            n4 = n
            n4_not_found = False
        if not n1_not_found and not n4_not_found: break
    return n1, n4

def run_prefixsum_based_inverleaved_sum_pruning(cur, table_name, attribute_name, N, a, b):
    create_sorted_table(cur, table_name, attribute_name)

    n1_not_found = n4_not_found = True
    i = 0
    j = N-1
    prefix_sum = inverse_prefix_sum = 0
    #n1 = n4 = None
    while i<j:
        prefix_sum += get_from_sorted_table(cur, table_name, attribute_name, i+1)
        inverse_prefix_sum += get_from_sorted_table(cur, table_name, attribute_name, j+1)
        #print inverse_prefix_sum
        if inverse_prefix_sum >= a and n1_not_found:
            n1 = i+1
            n1_not_found = False
        if prefix_sum > b and n4_not_found:
            n4 = i
            n4_not_found = False
        if not n1_not_found and not n4_not_found: break
        i += 1
        j -= 1
    return n1, n4


def get_cardinality_pruning_gain(search):
    # n_subsets_entire = float(n_of_subsets(paql_eval.N, 0, paql_eval.N))
    n_subsets_entire = search.search_space_size
    n_subsets_pruned = float(n_of_subsets(search.N, search.lb, search.ub))
    return 1.0 - (n_subsets_pruned / n_subsets_entire)



class Interval():
    def __init__(self,a=None,b=None):
        self.a = a
        self.b = b

    def __str__(self):
        return "[{}, {}]".format(self.a, self.b)

    def intersection(i1, i2):
        debug("INTERSECTION: %s %s", i1, i2)
        if i1 is None and i2 is None:
            return Interval()

        if i1.a <= i2.a:
            if i1.b < i2.a: # no overlap
                return Interval()
            elif i2.b > i1.b:
                return Interval(i2.a, i1.b)
            else:
                return Interval(i2.a, i2.b)
        else:
            if i2.b < i1.a: # no overlap
                return Interval()
            elif i1.b > i2.b:
                return Interval(i1.a, i2.b)
            else:
                return Interval(i1.a, i1.b)

    def extremes(self):
        return self.a, self.b


def intersect_cardinality_bounds(db, core_table, aggrs, N, prunetype):
    """
    # aggrs is a dicionary of aggregations and their bounds, i.e.
    # {'count': {u'>=': 0.0}, u'sum_price': {u'<=': 2000.0, u'>=': 300.0}}

    aggrs is a dictionary of aggregations and their bounds and attributes,
    e.g. {'count': (10, 20), 'sum': (100, 200, 'price')}

    This function converts every global constraint into a cardinality constraint.
    NOTE: so far, only count() and sum() are supported

    It returns an overall cardinality constraint which comes from the
    interval intersection of all defined constraints.
    """
    # Handle "default" prunetype, when prunetype is just True
    if isinstance(prunetype, bool):
        if prunetype:
            prunetype = "minmax"

    aggr_bounds = Interval(0, N)
    for (agg, attr), (lb, ub) in aggrs.iteritems():
        if agg == "count" and attr == "*":
            lb = int(lb) if float(lb) != float(-sys.maxint) else 0
            ub = int(ub) if float(ub) != float(+sys.maxint) else N
        elif agg == "sum":
            if prunetype=='minmax' or prunetype=='default' or prunetype is None:
                lb, ub = run_minmax_based_sum_pruning(db, core_table._core_table_name, attr, N, lb, ub)
            elif prunetype=='prefsum':
                lb, ub = run_prefixsum_based_sum_pruning(db, core_table._core_table_name, attr, N, lb, ub)
            elif prunetype=='prefsum2':
                lb, ub = run_prefixsum_based_inverleaved_sum_pruning(db, core_table._core_table_name, attr, N, lb, ub)
            else:
                raise Exception('prunetype %s not recognized' % prunetype)
        else:
            raise Exception("Aggregation '{}' not supported.".format(agg))
        debug("%s %s %s", lb, agg, ub)
        aggr_bounds = aggr_bounds.intersection(Interval(lb, ub))

    debug("FINAL CARDINALITY BOUNDS: %s", aggr_bounds.extremes())
    return aggr_bounds.extremes()



def compute_N(db, param):
    pass



if __name__=='__main__':
    db = DBConnection()

    N = compute_N(db, 'Stock')
    a = 1000
    b = 2000

    # This is a sort of materialized view, very fastly computed by postgres
    # It is temporary, as it will be destroyed at the end of the session
    print "Create (temporary) sorted table"
    start = time.time()
    create_sorted_table(db.cur, 'Stock', 'price')
    print time.time() - start, "seconds"

    print
    print "Loose bounds based on min and max"
    start = time.time()
    n1,n4 = run_minmax_based_sum_pruning(db, 'Stock', 'price', N, a, b)
    print time.time() - start, "seconds"
    print "n1=", n1
    print "n4=", n4

    print
    print "Strict bounds based on prefix sums (pre-computed)"
    start = time.time()
    n1,n4 = run_prefixsum_based_sum_pruning(db.cur, 'Stock', 'price', N, a, b)
    print time.time() - start, "seconds"
    print "n1=", n1
    print "n4=", n4

    print
    print "Strict bounds based on prefix sums (inverleaved computation of prefix sums)"
    start = time.time()
    n1,n4 = run_prefixsum_based_inverleaved_sum_pruning(db.cur, 'Stock', 'price', N, a, b)
    print time.time() - start, "seconds"
    print "n1=", n1
    print "n4=", n4

    db.close()






