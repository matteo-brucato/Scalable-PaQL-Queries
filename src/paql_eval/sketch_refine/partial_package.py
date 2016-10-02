import random
from copy import copy

from src.data_model.memory_representation.__package import Package
from src.data_model.memory_representation.sequence_based import SequenceBasedPackage
from src.data_model.tuple import Repr, Tuple
from src.paql.aggregates import SumAggr, CountAggr
from src.utils.log import log



class PartialPackage(object):
    """
    This is an assignment to each cluster. For a cluster, the assignment is either to the representatives of that
    cluster, or to the actual tuples.
    If the PartialSolution has an assingment to actual tuples from all clusters, then the PartialSolution is also
    a complete solution, in which case it can be transformed into a Package.
    """

    def __init__(self, reduced_space):
        self.reduced_space = reduced_space

        self.sol = {
            "R": { cid: [] for cid in self.reduced_space.cids },
            "O": { cid: [] for cid in self.reduced_space.cids },
        }

        self.sum = {
            "R": { cid: { attr: 0.0 for attr in self.reduced_space.query_attrs }
                   for cid in self.reduced_space.cids },
            "O": { cid: { attr: 0.0 for attr in self.reduced_space.query_attrs }
                   for cid in self.reduced_space.cids },
        }

        self.count = {
            "R": { cid: 0 for cid in self.reduced_space.cids },
            "O": { cid: 0 for cid in self.reduced_space.cids },
        }

        # Initially, the set of completed cids is only given by clusters with zero tuples in them
        self.completed_cids = { cid for cid, n_tuples in self.reduced_space.n_tuples_per_cid.iteritems()
                                if n_tuples==0 }

        # Initially, no cluster is set to infeasible. This dictionary will contain infeasible constraints per each cid
        self.infeasible_cids = { }


    def __copy__(self):
        pscopy = PartialPackage(self.reduced_space)
        pscopy.sol = {
            "R": { cid: list(self.sol["R"][cid]) for cid in self.reduced_space.cids },
            "O": { cid: list(self.sol["O"][cid]) for cid in self.reduced_space.cids },
        }
        pscopy.sum = {
            "R": { cid: { attr: self.sum["R"][cid][attr] for attr in self.reduced_space.query_attrs }
                   for cid in self.reduced_space.cids },
            "O": { cid: { attr: self.sum["O"][cid][attr] for attr in self.reduced_space.query_attrs }
                   for cid in self.reduced_space.cids },
        }
        pscopy.count = {
            "R": { cid: self.count["R"][cid] for cid in self.reduced_space.cids },
            "O": { cid: self.count["O"][cid] for cid in self.reduced_space.cids },
        }
        pscopy.completed_cids = copy(self.completed_cids)
        pscopy.infeasible_cids = copy(self.infeasible_cids)
        return pscopy


    def __str__(self):
        solved_original_space_cids = self.get_solved_original_cids()
        solved_reduced_space_cids = self.get_solved_reduced_cids()
        _str = ["PartialPackage = {"]
        for cid in self.reduced_space.cids:
            if cid in solved_original_space_cids:
                _str.append("\tCluster {}:\tTake {} tuples{}".format(
                    cid, sum(int(s) for t, s in self.sol["O"][cid]),
                    " (*)" if cid in self.completed_cids else ""))
            elif cid in solved_reduced_space_cids:
                _str.append("\tCluster {}:\tTake {} representatives".format(
                    cid, sum(int(s) for t, s in self.sol["R"][cid]), ))
            elif cid in self.infeasible_cids:
                _str.append("\tCluster {}:\tInfeasible".format(cid))
            else:
                _str.append("\tCluster {}:\tYet unsolved".format(cid))
        _str.append("}")
        return "\n".join(_str)


    @property
    def is_infeasible(self):
        # True if there is at least one infeasible cluster (except None, which is not a proper cluster id)
        if None in self.infeasible_cids:
            return len(self.infeasible_cids) > 1
        else:
            return len(self.infeasible_cids) > 0


    @property
    def is_complete(self):
        return all(cid in self.completed_cids for cid in self.reduced_space.n_tuples_per_cid)


    def is_cluster_infeasible(self, cid):
        response = cid in self.infeasible_cids
        assert not response or self.is_infeasible
        return response


    def set_infeasible(self, cid, infeasible_constraints):
        self.infeasible_cids[cid] = infeasible_constraints


    def clear_reduced_space_solution_for_cluster(self, cid):
        del self.sol["R"][cid][:]
        self.count["R"][cid] = 0
        for attr in self.reduced_space.query_attrs:
            self.sum["R"][cid][attr] = 0
        assert cid not in self.get_solved_reduced_cids()


    def clear_original_space_solution_for_cluster(self, cid):
        del self.sol["O"][cid][:]
        self.count["O"][cid] = 0
        for attr in self.reduced_space.query_attrs:
            self.sum["O"][cid][attr] = 0
        assert cid not in self.get_solved_original_cids()
        if cid in self.completed_cids and self.reduced_space.n_tuples_per_cid[cid] > 0:
            self.completed_cids.remove(cid)


    def add_reduced_space_solution_for_cluster(self, cid, r, sol_val):
        assert isinstance(r, Repr)

        # NOTE: Only store the tuple id's, not the entire data
        stored_r = Repr(attrs=["cid"], record=r)
        self.sol["R"][cid].append((stored_r, sol_val))

        # Update stored aggregates
        self.count["R"][cid] += sol_val
        for attr in self.reduced_space.query_attrs:
            self.sum["R"][cid][attr] += getattr(r, attr) * sol_val


    def add_original_space_solution_for_cluster(self, cid, t, sol_val):
        assert t is None or isinstance(t, Tuple), t

        # NOTE: For now only support proper subset packages
        assert 0 <= round(sol_val) <= 1, "sol_val = {}".format(sol_val)

        sol_val = round(sol_val)

        if t is not None:
            # NOTE: Only store the tuple id's, not the entire data
            stored_t = Tuple(attrs=["id"], record=t)
            self.sol["O"][cid].append((stored_t, sol_val))

            # Update stored aggregates
            self.count["O"][cid] += sol_val
            for attr in self.reduced_space.query_attrs:
                self.sum["O"][cid][attr] += getattr(t, attr) * sol_val

        else:
            # NOTE: Only store the tuple id's, not the entire data
            self.sol["O"][cid].append((None, sol_val))

        # Check if cluster is completed
        if len(self.sol["O"][cid])==self.reduced_space.n_tuples_per_cid[cid]:
            assert cid not in self.completed_cids
            # print ">> BEFORE", self.completed_cids
            self.completed_cids.add(cid)

        elif len(self.sol["O"][cid]) >= self.reduced_space.n_tuples_per_cid[cid]:
            print len(self.sol["O"][cid]), ">=", len(self.reduced_space.clusters[cid])
            for s in self.sol["O"][cid]:
                print s
            for t in self.reduced_space.clusters[cid]:
                print t.id
            raise AssertionError("More assignments than tuples to cluster {}.".format(cid))


    def get_count(self, cid):
        if len(self.sol["O"][cid]) > 0:
            return self.count["O"][cid]
        else:
            return self.count["R"][cid]


    def get_sum(self, cid, attr):
        if len(self.sol["O"][cid]) > 0:
            return self.sum["O"][cid][attr]
        else:
            return self.sum["R"][cid][attr]


    def get_cluster_sol(self, cid):
        if cid is None:
            return []

        # If the cluster has been solved in original space, return the original space solution
        if len(self.sol["O"][cid]) > 0:
            return [(t, v) for t, v in self.sol["O"][cid]]

        # Otherwise, if it was solved in the reduced space, return the reduced space solution
        if len(self.sol["R"][cid]) > 0:
            return [(r, v) for r, v in self.sol["R"][cid]]

        # Otherwise, return the empty solution
        return tuple()


    def get_solved_reduced_cids(self):
        # Return all cluster ids for which there is a non-empty solution in the reduced space, that is, clusters that
        # have been solved in the reduced space
        return set(cid for cid, sol in self.sol["R"].iteritems() if len(sol) > 0)


    def get_solved_original_cids(self):
        # Return all cluster ids for which there is a non-empty solution in the original space, that is, clusters that
        # have been solved in the original space
        return set(cid for cid, sol in self.sol["O"].iteritems() if len(sol) > 0)


    def to_candidate_package(self):
        """
        Convert a complete (or partial) solution into a Package instance by including the original tuples
        that are included in the partial package
        """
        log("Identifying candidate id's...")
        candidate_ids = []
        for cid, sols in self.sol["O"].iteritems():
            for t, sol in sols:
                # If it says 1, include the id
                if round(sol) == 1:
                    candidate_ids.append(t.id)

                # Otherwise, just skip this tuple
                elif round(sol) == 0:
                    pass
                else:
                    raise Exception("Solution value unexpected (neither 0 or 1): {}".format(sol))

        # Construct the Package from the tuple ids
        log("Creating SequenceBasedPackage from candidate id's...")
        candidate = SequenceBasedPackage.from_ids(self.reduced_space, candidate_ids)
        return candidate


    def to_random_candidate_package(self):
        # Convert a complete (or partial) solution into a Package instance
        # by picking a random tuple for each representative in the partial package

        candidate_ids = []
        for cid in self.reduced_space.cids:
            for tr, sol in self.get_cluster_sol(cid):
                if len(self.sol["O"][cid]) > 0:
                    # If it says 1, include the id
                    if round(sol)==1:
                        candidate_ids.append(tr.id)
                    # Otherwise, just skip this tuple
                    elif round(sol)==0:
                        pass
                    else:
                        raise Exception("Solution value unexpected (neither 0 or 1): {}".format(sol))
                else:
                    # Pick as many random tuples from r's group as its multiplicity in the solution
                    sample = random.sample(self.sol["O"][cid], min(int(round(sol)), len(self.sol["O"][cid])))

                    for t, tsol in sample:
                        candidate_ids.append(t.id)

        # Construct the Package from the tuple ids
        candidate = Package.from_ids(self.reduced_space, candidate_ids)
        return candidate


    def get_partial_objective_value(self):
        """
        Returns an objective value that is computed by also considering the representative tuples in case a cluster
        is not solved in the original space.
        """
        if self.is_infeasible:
            raise AssertionError("This partial solution contains some infeasible clusters.")

        obj_aggr = self.reduced_space.query.objective.get_aggregate()
        obj_attrs = obj_aggr.args
        if len(obj_attrs) > 1:
            raise Exception("For now we only support one attribute in objective function.")
        obj_attr = obj_attrs[0]

        solved_original_cids = self.completed_cids
        solved_reducedonly_cids = self.get_solved_reduced_cids() - solved_original_cids
        empty_cids = set(self.reduced_space.n_tuples_per_cid) - solved_original_cids - solved_reducedonly_cids
        assert len(empty_cids)==0, "Can't compute objective value is some clusters have not been solved yet."

        vals = []
        for cid in solved_original_cids | solved_reducedonly_cids:
            for t, sol in self.get_cluster_sol(cid):
                v = int(round(sol))
                # If it says 1, include its objective attribute value
                if v >= 1:
                    if obj_attr!="*":
                        vals += [getattr(t, obj_attr)] * v
                    else:
                        vals += [1] * v
                # Otherwise, just skip this tuple
                elif v==0:
                    pass
                else:
                    raise Exception("Solution value unexpected (neither 0 or 1): {}".format(sol))

        if isinstance(obj_aggr, SumAggr):
            obj_val = sum(vals)
        elif isinstance(obj_aggr, CountAggr):
            obj_val = len(vals)
        else:
            raise Exception("For now we only support sum or count objective function.")

        return obj_val
