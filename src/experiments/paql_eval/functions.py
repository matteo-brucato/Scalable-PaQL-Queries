import heapq
import logging
import math

import numpy as np
import scipy

from src.data_model.memory_representation.__package import Package
from src.paql_eval.sketch_refine.greedy_backtracking import GreedyBacktrackRunInfo
from src.utils.log import debug
from src.utils.utils import avgp, pretty_table_str_named



def is_better(alg, opt, q):
    assert q.objective is not None
    obj_type = q.objective["type"]
    if obj_type=="maximize":
        return alg > opt
    elif obj_type=="minimize":
        return alg < opt
    else:
        raise Exception("Objective type '{}' not recognized!".format(obj_type))



def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func


    return decorate



def real_avgpoints(L, clust_attrs_indexes, choice, ratio=None):
    """
    Returns closest points to average point. The number of points is calculated as a ratio on the total number
    of points in L.
    """
    avgL = avgp(L)[0]

    if ratio is None:
        n_points_to_return = 1
    else:
        n_points_to_return = int(math.ceil(len(L) * ratio))

    if choice=="closest-to-avg":
        sign_factor = -1
    elif choice=="furthest-from-avg":
        sign_factor = 1
    else:
        raise Exception

    # NOTE: These points are not in any particular order
    closest_to_avg_heap = [
        (
            sign_factor * scipy.spatial.distance.euclidean(
                [L[i][j] for j in clust_attrs_indexes],
                [avgL[j] for j in clust_attrs_indexes]
            ),
            L[i])
        for i in xrange(0, n_points_to_return)
        ]
    heapq.heapify(closest_to_avg_heap)
    print "BEFORE"
    print "\n".join("dist={}\t{}".format(sign_factor * d, l) for d, l in sorted(closest_to_avg_heap))
    for i in xrange(n_points_to_return, len(L)):
        l = L[i]
        dist_l = scipy.spatial.distance.euclidean(
            [l[j] for j in clust_attrs_indexes],
            [avgL[j] for j in clust_attrs_indexes]
        )

        # Take furthest-away-from-average point from the closest-to-average points
        max_dist, furthest_to_avg = min(closest_to_avg_heap)

        if (sign_factor==-1 and dist_l < sign_factor * max_dist) or (
                        sign_factor==1 and dist_l > sign_factor * max_dist):
            # Remove furthest_to_avg and replace it with l
            assert l!=furthest_to_avg and max_dist!=dist_l
            a, b = heapq.heappop(closest_to_avg_heap)
            assert a==max_dist and b==furthest_to_avg
            heapq.heappush(closest_to_avg_heap, (sign_factor * dist_l, l))
        else:
            pass
        assert len(closest_to_avg_heap)==n_points_to_return

    print "AFTER"
    print "\n".join("dist={}\t{}".format(sign_factor * d, l) for d, l in sorted(closest_to_avg_heap))

    result_points = [el for dist, el in closest_to_avg_heap]

    try:
        assert set(result_points).issubset(set(L))
    except AssertionError as e:
        print L
        print result_points
        for p in result_points:
            if not p in L:
                print "{} not in L".format(p)
        raise e

    return result_points



def performance_ratio(opt, alg, q):
    if opt==alg:
        # This also covers the case opt=0 and alg=0
        return 1.0
    if np.sign(opt)!=np.sign(alg):
        # If opt and alg have different signs, the performance ratio is not well defined
        return float("nan")
    if q.objective["type"]=="maximize":
        return opt / alg if alg!=0 else float("nan")
    elif q.objective["type"]=="minimize":
        return alg / opt if opt!=0 else float("nan")



def check_candidate(cand, candidate_obj_val, perf_ratio, compare_to_optimal,
                    rs_search, opt_p, opt_obj_val):
    assert isinstance(cand, Package)
    is_feasible_candidate = cand.is_valid()
    print "Candidate:"
    if logging.getLogger().getEffectiveLevel()==logging.DEBUG:
        debug(pretty_table_str_named(
            cand.iter_tuples(), headers=["id"] + rs_search.all_attrs))
    if compare_to_optimal:
        print "Actual optimal objective value is: {}".format(opt_obj_val)
        print "Candiate objective value is: {}".format(candidate_obj_val)
        print "Deviation from optimal objective value: {:.3f}%".format(
            100 * abs((opt_obj_val - candidate_obj_val) / opt_obj_val)
            if opt_obj_val!=0 else (opt_obj_val - candidate_obj_val))
        print "Performance Ratio: {}".format(perf_ratio)
        if cand > opt_p:
            print "========= ASSERTION WARNING! ==========="
            print "Actual optimal package is not better than or qual to candidate package!"
            print "opt_obj_val={}; candidate_obj_val={}".format(
                opt_obj_val, candidate_obj_val)

    try:
        assert is_feasible_candidate
    except AssertionError as assertion_e:
        assert isinstance(opt_p, Package)
        assert isinstance(cand, Package)
        print "========= ASSERTION ERROR! ==========="
        print "Query was:\n{}".format(rs_search.query)
        print "Solution returned by full backtrack is not feasible!"
        print "Candidate global vals ({}):\n{}".format(
            ["{}({})".format(aggr, attr) for aggr, attr in rs_search.query.coalesced_gcs.iterkeys()],
            cand.get_coalesced_global_scores())
        print "Optimal global vals ({}):\n{}".format(
            ["{}({})".format(aggr, attr) for aggr, attr in rs_search.query.coalesced_gcs.iterkeys()],
            opt_p.get_coalesced_global_scores())
        print "Optimal combo:", opt_p.combination
        print "Optimal table:\n", pretty_table_str_named(
            opt_p.iter_tuples(), headers=["id"] + rs_search.clust_attrs)
        print "Candidate combo:", cand.combination
        print "Candidate table:\n", pretty_table_str_named(
            cand.iter_tuples(), headers=["id"] + rs_search.clust_attrs)

        print opt_p.search
        print cand.search

        raise assertion_e



def print_runinfo_backtracking_algo(backtrack_runinfo, opt_obj_val, compare_to_optimal, rs_search,
                                    opt_wallclock_time, recluster, use_index):
    assert isinstance(backtrack_runinfo, GreedyBacktrackRunInfo)
    print "-" * 50
    print backtrack_runinfo
    print "All augmenting problems ({}):\n{}".format(
        len(backtrack_runinfo.augmenting_problems_info),
        "\n".join("  {}. Cid={},\t Complete? {},\t Perf ratio: {}".format(
            i, infos["cid"], infos["partial-sol"].is_complete,
            performance_ratio(opt_obj_val, infos["partial-sol"].get_partial_objective_value())
            if compare_to_optimal else "--"
        ) for i, infos in enumerate(backtrack_runinfo.augmenting_problems_info)))
    print "-" * 50
    print ">> INFO BACKTRACKING:"
    print ">> DATASET SIZE:", rs_search.N
    print ">> N CLUSTERS:", rs_search.n_clusters
    print ">> N RECURSIVE CALLS:",\
        backtrack_runinfo.n_recursive_calls
    print ">> N SOLVED PROBLEMS:",\
        len(backtrack_runinfo.cplex_run_info)
    print ">> MAX PROBLEM SIZE :",\
        max(stat.cplex_problem_size for stat in backtrack_runinfo.cplex_run_info)
    print ">> N INFEASIBLE AUGMENTING PROBLEMS :",\
        backtrack_runinfo.n_infeasible_augmenting_problems
    print ">> SUM CPLEX WALLCLOCK TIME:",\
        sum(stat.cplex_wallclock_time for stat in backtrack_runinfo.cplex_run_info)
    print ">> BACKTRACKING HEURISTIC SEARCH WALLCLOCK TIME:",\
        backtrack_runinfo.total_wallclock_time
    if compare_to_optimal:
        print ">> OPTIMAL SEARCH WALLCLOCK TIME:", opt_wallclock_time
    if use_index:
        print "RECOVER PARTITIONING USING INDEX, TIME:",\
            rs_search.clustering_recover_w_index_wc_time
    if recluster:
        print "CLUSTERING TIME:",\
            rs_search.clustering_online_wc_time
