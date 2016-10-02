import gc
import hashlib
import itertools
import logging
import math
import sys
import traceback
from logging import warning, debug

import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, CPLEX, LpStatus

from src.dbms.utils import sql_get_all_attributes, sql_table_column_data_type
from src.paql.constraints import *
from src.paql.expression_trees.expression_trees import ArithmeticExpression
from src.paql.expression_trees.syntax_tree import Expression
from src.paql.objectives import *
from src.utils.utils import op_to_opstr



class NotPackageQueryException(Exception):
    pass


class PaQLParserError(Exception):
    pass






class PackageQuery(object):
    allowed_dbms_data_types = {
        "integer",
        "bigint",
        "double precision",
        # "numeric",
        # "numeric(15,2)"
    }

    @property
    def table_name(self):
        assert len(self.rel_namespace.values()) == 1
        return self.rel_namespace.itervalues().next()


    @table_name.setter
    def table_name(self, table_name):
        assert len(self.rel_namespace.values()) == 1
        if self.table_name is not None and self.rel_namespace is not None:
            for rel, relname in self.rel_namespace.iteritems():
                if relname.lower() == self.table_name.lower():
                    self.rel_namespace[rel] = table_name

        self._paql_query_str_stale = True


    @property
    def bc_query(self):
        bc_query = "SELECT * FROM {}".format(
            ','.join([
                rel_name + " " + rel_alias for rel_alias, rel_name in self.rel_namespace.iteritems()
            ]))
        where_clause_str = self.where_expr.get_str()
        if where_clause_str:
            bc_query += " WHERE {}".format(where_clause_str)
        if self.limit is not None and self.limit["TYPE"] =="INPUT":
            bc_query += " LIMIT {}".format(self.limit["LIMIT"])
        return bc_query


    def __init__(self, d):
        assert isinstance(d, dict)

        self._paql_query_str = None
        self._paql_query_str_stale = True

        # self.package_rel_names = d["package rel names"]
        self.rel_namespace = d["namespace"]
        self.rel_repeats = d["repeats"]
        self.where_expr = d["where expr"]
        self.such_that_expr = d["such that expr"]

        if d["objective expr"] is not None:
            self.objective = PackageQueryObjective(
                sqlquery_expr=d["objective expr"].get_sql_arithmetic_expression(),
                sense=d["objective sense"])
        else:
            self.objective = None

        self.limit = d["limit"]

        # NOTE: For now, assuming that the query is single-table.
        # TODO: We need to take into account REPEAT! It's not implemented yet!
        # rel_names = self.rel_namespace.values()
        assert len(self.rel_namespace.values()) == 1, "Not a single-table package query!"
        # self.table_name = self.bc_query.lower().split("from")[1].split("where")[0].split()[0].strip()
        # self.table_name = rel_names[0]


    def __str__(self):
        raise NotImplementedError


    def md5(self):
        return hashlib.md5(str(self)).hexdigest()


    @classmethod
    def get_json_from_paql(cls, paql_str):
        from subprocess import Popen, PIPE

        p = Popen(["PaQL_Parser"], stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
        json_str, err = p.communicate(input=paql_str)
        p.wait()

        if err != "":
            raise PaQLParserError(err)

        return json_str


    @classmethod
    def from_paql(cls, paql_str):
        """
        Returns a new PackageQuery object from a PaQL query string. This is the method that you would call more often.
        :param paql_str: A string containing a PaQL query
        :rtype : PackageQuery
        """
        json_str = PackageQuery.get_json_from_paql(paql_str)

        try:
            package_query = cls.from_json(json_str)
        except ValueError as e:
            traceback.print_exc(file=sys.stdout)
            raise PaQLParserError(e)
        else:
            package_query._paql_query_str = paql_str
            package_query._paql_query_str_stale = False
            return package_query
        finally:
            gc.collect()


    @classmethod
    def from_json(cls, json_str):
        """
        Returns a new PackageQuery object from a JSON string. This method is usually called by from_PaQL() to
        transform the paql parser output (which is a JSON) into a PackageQuery object.
        This is the main entry point from the direct output of the paql parser.
        :param json_str: A string containing a JSON structure for a parsed PaQL query
        """
        import json

        q = json.loads(json_str)

        # The namespace of relations defined by the query. A dictionary alias -> relation-name.
        # This way, all references to relations can be just made based on the alias names, and we can avoid confusion
        # when nested queries contain the same relation names, etc.
        rel_namespace = { }

        # The mapping from relation aliases into their corresponding REPEAT values.
        rel_repeats = { }

        # The list of relation aliases which form the PACKAGE.
        package_rel_names = []

        # TODO: Ideally, if the query is not a package query we may want to just execute it as it is...
        # TODO: If it doesn't contain the PACKAGE clause, we should make sure it does not contain SUCH THAT either.

        # Check if it's package query and store reference to relation names
        for select_item in q["SELECT"]:
            assert type(select_item) == dict

            if select_item["NODE_TYPE"] == "*":
                raise NotPackageQueryException()

            elif select_item["NODE_TYPE"] == "COL_REF":
                raise NotPackageQueryException()

            elif select_item["NODE_TYPE"] == "PACKAGE":
                package_rel_names.extend(r["REL_NAME"] for r in select_item["PACKAGE_RELS"])

            else:
                raise Exception("Problem in SELECT clause, NODE_TYPE non recognized: " + select_item["NODE_TYPE"])

        # Store relation names and aliases, and repeat constraint for each of them
        # These are stored in a dictionary rel_namespace(key=rel_alias, val=rel_names)
        for from_ in q["FROM"]:
            assert type(from_) == dict
            rel_name = from_["REL_NAME"]
            rel_alias = from_.get("REL_ALIAS", rel_name)
            repeat = from_.get("REPEAT", -1)
            rel_namespace[rel_alias] = rel_name
            rel_repeats[rel_alias] = repeat

        # Make sure that all relation aliases referred in PACKAGE(...) are in the FROM clause as well
        assert all(p_rel_name in rel_namespace for p_rel_name in package_rel_names)

        # Stricter (for now): Make sure that they are exactly the same relation references
        assert set(package_rel_names) == set(rel_namespace.iterkeys())

        # Create WHERE clause expression tree
        where_clause = Expression(q["WHERE"])

        # Create SUCH THAT clause expression tree
        such_that_clause = Expression(q["SUCH-THAT"])

        # Create objective clause expression tree
        if q["OBJECTIVE"] is not None:
            objective_expr = Expression(q["OBJECTIVE"]["EXPR"])

            if q["OBJECTIVE"]["TYPE"] == "MAXIMIZE":
                # objective = { "type": "maximize", "expr": objective_expr }
                objective_sense = ObjectiveSenseMAX()

            elif q["OBJECTIVE"]["TYPE"] == "MINIMIZE":
                # objective = { "type": "minimize", "expr": objective_expr }
                objective_sense = ObjectiveSenseMIN()

            else:
                raise Exception("Unsupported objective type: `{}'".format(q["OBJECTIVE"]["TYPE"]))

        else:
            objective_expr = objective_sense = None

        query_dict = {
            # "package rel names": package_rel_names,
            "where expr": where_clause,
            "such that expr": such_that_clause,
            "objective expr": objective_expr,
            "objective sense": objective_sense,
            "namespace": rel_namespace,
            "repeats": rel_repeats,
            "limit": q["LIMIT"],
        }

        if such_that_clause.is_conjunctive() and where_clause.is_conjunctive():
            return ConjunctivePackageQuery(query_dict)
        else:
            return cls(query_dict)


    @staticmethod
    def from_uncoalesced_constraints(table_name, unc_bcs, unc_gcs, objective):
        """
        This method creates a new PackageQuery from sets of uncoalesced constraints and an objective.
        """
        bc_query = "SELECT * FROM {} {}".format(table_name, "WHERE true" if len(unc_bcs) > 0 else "")
        for attr, op, n in unc_bcs:
            bc_query += " AND {a} {o} {b}".format(a=attr, o=op_to_opstr(op), b=n)

        gc_queries = []
        gc_ranges = []
        for (aggr, attr), op, n in unc_gcs:
            gc_query = "SELECT {aggr}({attr}) FROM memory_representations".format(aggr=aggr, attr=attr)
            if op == operator.le:
                # gc_range = (-sys.maxint, n)
                gc_range = (-float("inf"), n)
            elif op == operator.ge:
                # gc_range = (n, sys.maxint)
                gc_range = (n, float("inf"))
            elif op == operator.eq:
                gc_range = (n, n)
            else:
                raise Exception("Operator '{}' not supported yet.".format(op))
            gc_queries.append(gc_query)
            gc_ranges.append(gc_range)

        return PackageQuery({
            "bc": bc_query,
            "gc": map(lambda x: (x[0], x[1][0], x[1][1]), zip(gc_queries, gc_ranges)),
            "objective": objective,
        })


    def get_objective_attributes(self):
        attrs = set()
        if self.objective is not None:
            for attr in self.objective.get_attributes():
                if attr != "*":
                    attrs.add(attr)
        return attrs


    def get_bcs_attributes(self):
        return set(attr for attr in self.coalesced_bcs) - {"*"}


    def get_gcs_attributes(self):
        gcs_attrs = set()
        for gc in self.coalesced_gcs:
            assert isinstance(gc, CGlobalConstraint)

            gcs_attrs.update(gc.get_attributes())

        return gcs_attrs


    def get_attributes(self):
        # FIXME: If this is a relaxed query, you should return all attributes including those of the original query.
        return self.get_bcs_attributes() | self.get_gcs_attributes() | self.get_objective_attributes()


    def get_data_attributes(self, db):
        all_data_attributes = sql_get_all_attributes(db, self.table_name)

        # Only pick the data attributes of the allowed data type
        data_attributes = set()
        for data_attr in all_data_attributes:
            attribute_type = sql_table_column_data_type(db, self.table_name, data_attr)
            if attribute_type in self.allowed_dbms_data_types:
                data_attributes.add(data_attr)
        return sorted(data_attributes)



    def get_paql_str(self, redo=False, recompute_gcs=True, coalesced=False):
        raise NotImplementedError


    def abs_ugc_errors(self, gc_scores, attrs=None):
        """
        Returns absolute errors for each (uncoalesced) global constraint.
        """
        if attrs is None:
            use_attrs = self.get_attributes()
        else:
            use_attrs = set(attrs)

        return {
            (aggr, attr): max(0, c - gc_scores[aggr, attr] if op == operator.ge else gc_scores[aggr, attr] - c)
            for (aggr, attr), op, c in self.uncoalesced_gcs if attr == "*" or attr in use_attrs
        }


    def error_mape(self, u_gc_scores, u_bc_scores):
        errorsum = .0
        n_gcs = 0
        n_bcs = 0

        for i, ((aggr, attr), op, c) in enumerate(self.uncoalesced_gcs):
            score = u_gc_scores[i]
            if not op(score, c):
                errorsum += abs((c - score) / c)
            n_gcs += 1

        for bscores in u_bc_scores:
            for i, (attr, op, c) in enumerate(self.uncoalesced_bcs):
                score = bscores[i]
                if not op(score, c):
                    errorsum += abs((c - score) / c)
                n_bcs += 1

        if n_gcs + n_bcs > 0:
            return errorsum / (n_gcs + n_bcs)

        else:
            assert errorsum == 0
            return 0


    def generate_data_for_selectivity(self, selectivity, n_tuples):
        """
        NOTE: This is currently unused. Not even sure if I completed it. But give a look at it again because
        there were some interesting ideas.
        """
        def generate_valid_and_invalid_subsets(n_vars, n_subsets, n_valid):
            # TODO: Read again this function. There's some interesting logic
            n_subsets = int(math.ceil(n_subsets))
            n_valid = int(math.ceil(n_valid))

            assert n_valid <= n_subsets == 2**n_vars

            valid = []
            invalid = []

            # This must be always valid (it is the sum of no tuples)
            # valid.append( (0,)*n_vars )
            valid.append(0)

            # Generate half of vars valid and half invalid
            for i in range(n_vars):
                if len(valid) < n_valid/2.:
                    # valid.append(tuple( bit for bit in ('{:0%dbms}' % n_tuples).format(2**i) ))
                    valid.append(2**i)
                elif len(invalid) < (n_subsets - n_valid)/2.:
                    # invalid.append(tuple( bit for bit in ('{:0%dbms}' % n_tuples).format(2**i) ))
                    invalid.append(2**i)
                else:
                    valid.append(2**i)

            # Generate more invalid (up to n_subsets-n_valid) by combining invalid + invalid
            while len(invalid) < n_subsets-n_valid:
                found = False
                for i in range(len(invalid)):
                    for j in range(len(invalid)):
                        new_invalid = invalid[i] | invalid[j]
                        if new_invalid not in invalid:
                            invalid.append(new_invalid)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break

            # If more invalid are needed, generate them by combining invalid + valid
            while len(invalid) < n_subsets-n_valid:
                found = False
                for i in range(len(invalid)):
                    for j in range(len(valid)):
                        new_invalid = invalid[i] | valid[j]
                        if new_invalid not in invalid:
                            invalid.append(new_invalid)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    raise Exception

            # All the remaining ones are valid
            valid = set(range(n_subsets)) - set(invalid)

            assert len(valid) == n_valid
            assert len(valid) + len(invalid) == n_subsets

            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                debug("n invalid = {}".format(n_subsets - n_valid))
                debug("{}".format(valid))
                debug("{}".format(invalid))
                debug("{}".format([ tuple( bit for bit in ('{:0%dbms}' % n_tuples).format(i) ) for i in valid ]))
                debug("{}".format([ tuple( bit for bit in ('{:0%dbms}' % n_tuples).format(i) ) for i in invalid ]))

            return valid, invalid


        def generate_set_of_problems(base_prob, vars, total_n_constraints, n_valid_constraints, a, b):
            problems = []

            for valid in itertools.combinations(range(total_n_constraints), int(math.ceil(n_valid_constraints))):
                valid = set(valid)

                invalid = set(range(total_n_constraints)) - valid
                assert set(valid) | invalid == set(range(total_n_constraints))

                # The empty package must always be valid. TODO: Really?
                # valid = [0] + list(valid)

                prob = base_prob.copy()

                # valid = generate_valid_and_invalid_subsets(n_tuples, total_n_constraints, n_valid_constraints)[0]
                # valid = np.random.choice(range(total_n_constraints), size=n_valid_constraints, replace=False)
                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                    debug("VALID: {}".format(valid))
                    debug("INVALID: {}".format(sorted(set(range(total_n_constraints)) - set(valid))))

                # Add valid constraints to the problem
                n_valid_added = 0
                for i in valid:
                    package_bitmap = [ int(bit) for bit in ('{:0%dbms}' % n_tuples).format(i) ]
                    assert len(package_bitmap) == len(vars)

                    # Add a VALID constraint for this combination of tuples
                    prob += np.dot(vars, package_bitmap) >= a
                    prob += np.dot(vars, package_bitmap) <= b
                    n_valid_added += 1
                assert n_valid_added == len(valid)

                # Add invalid constraints to the problem
                n_invalid_added = 0

                if float(a) > -float("inf") and float(b) < float("inf"):
                    # In this case, we produce 2**(len(invalid)) new sub-problems, each for a different set of ways
                    # to break the constraints a <= sum() <= b
                    pairs_of_invalid_constraints = []
                    for i in invalid:
                        package_bitmap = [ int(bit) for bit in ('{:0%dbms}' % n_tuples).format(i) ]
                        pairs_of_invalid_constraints.append((
                            (package_bitmap, operator.le, a-1),
                            (package_bitmap, operator.ge, b+1),
                        ))

                    orig_prob = prob.copy()
                    for set_of_invalid in itertools.product(*pairs_of_invalid_constraints):
                        new_prob = orig_prob.copy()
                        for invalid_bitmap, op, c in set_of_invalid:
                            new_prob += op(np.dot(vars, invalid_bitmap), c)
                        problems.append(new_prob)

                else:
                    # In this case, we only generate one sub-problem by adding all invalid constraints
                    for i in invalid:
                        package_bitmap = [ int(bit) for bit in ('{:0%dbms}' % n_tuples).format(i) ]
                        assert len(package_bitmap) == len(vars)

                        # Add an INVALID (out of range) constraint for this combination of tuples
                        if float(a) > -float("inf") and float(b) < float("inf"):
                            raise Exception("Should never happen!")
                            # prob += np.dot(vars, package_bitmap) <= a-1
                        elif float(a) > -float("inf"):
                            prob += np.dot(vars, package_bitmap) <= a-1
                        elif float(b) < float("inf"):
                            prob += np.dot(vars, package_bitmap) >= b+1
                        else:
                            raise Exception
                    assert n_invalid_added == len(invalid)

                    problems.append(prob)

            return problems


        assert 0 <= selectivity <= 1
        assert n_tuples >= 0

        table_name_start = self.bc_query.lower().find("from ")
        table_name_end = self.bc_query[table_name_start+5:].lower().find(" ")
        table_name = self.bc_query[table_name_start+5:table_name_start+5+table_name_end]

        attribute_names = []
        ranges = []
        for j in range(len(self.gc_queries)):
            if 'sum(' in self.gc_queries[j].lower():
                attr_start = self.gc_queries[j].lower().find('sum(')
                attr_end = self.gc_queries[j][attr_start+4:].lower().find(')')
                attribute_names.append(self.gc_queries[j][attr_start+4:attr_start+4+attr_end])
                ranges.append(self.gc_ranges[j])
        debug("{} {}".format(attribute_names, ranges))
        assert len(attribute_names) == len(ranges)

        # Generate the data via CPLEX
        data_columns = []

        # Generate one column at a time. Each column is generated with a CPLEX problem
        for j in range(len(attribute_names)):
            a, b = ranges[j]

            total_n_constraints = 2**n_tuples
            n_valid_constraints = (1-selectivity) * total_n_constraints

            # Check satisfiability of requirements
            if n_valid_constraints == 0 and a <= 0 <= b:
                warning("Since a<=0<=b there is always at least one valid package, i.e. the empty package, "
                        "therefore selectivity=1 (where no package is valid) is impossible.")
                return None
            if n_valid_constraints == total_n_constraints and not a <= 0 <= b:
                warning("Since not a<=0<=b, the empty package may never be a valid package, "
                        "therefore selectivity=0 (where all packages are valid) is impossible.")
                return None

            # Create the base problem
            base_prob = LpProblem("package-builder", LpMinimize)
            base_prob += 0 # no objective

            # Add constraints to the problem
            vars = [
                LpVariable("{}_{}".format(attribute_names[j], i), -float("inf"), float("inf"), LpInteger)
                for i in range(n_tuples)
            ]

            # Generate all possible combination of problem constraints
            # One of them will be feasible and will give us the dataset
            problems = generate_set_of_problems(base_prob, vars, total_n_constraints, n_valid_constraints, a, b)

            # Now try to find one feasible problem
            for prob in problems:
                # Solve the problem
                debug("{}".format(prob))
                solver = CPLEX(msg=True, timeLimit=None)
                solver.solve(prob)

                # Check the problem status
                if LpStatus[prob.status]=='Infeasible':
                    debug("@@@@@@@@@@@@@@@@@ INFEASIBLE: CONTINUE")
                    continue

                elif LpStatus[prob.status]=='Undefined':
                    raise Exception("Problem is undefined.")

                elif LpStatus[prob.status]=='Optimal':
                    debug("################## OPTIMAL")
                    prob.roundSolution()
                    sol = [ v.varValue for v in prob.tuple_variables() if type(v.varValue) is float ]
                    data_columns.append(sol)
                    break

                else:
                    raise Exception("LP status: {}".format(LpStatus[prob.status]))

            else:
                raise Exception("Could not find feasible combination of constraints "
                                "for selectivity {} and {} tuples.".format(selectivity, n_tuples))

        tuples = np.array(data_columns).transpose()

        return table_name, attribute_names, tuples



class ConjunctivePackageQuery(PackageQuery):
    # TODO: later on, move the two staticmethods from_... outside. Make them just functions.
    # TODO: IMPORTANT! All base and gc queries MUST be instance of some class SQL_Query instead of just strings

    def __init__(self, query_dict):
        super(ConjunctivePackageQuery, self).__init__(query_dict)

        # Store the base and global constraints as coalesced and un-coalesced constraints
        gc_constraint_trees = []
        gc_ranges = []
        gcs = self.such_that_expr.get_ANDed_gc_list()
        for sqlquery_expr, gc_range_a, gc_range_b in gcs:
            if isinstance(sqlquery_expr, SQLQueryExpression):
                # Note: Technically, you'll get an expression tree of "constraint trees" (query plans). So you
                # should actually try to combine them into one single constraint tree. Right now I'm simplifying
                # by assuming that the expression tree is always a simple leaf (so directly a constraint tree).
                operator_tree_expr = sqlquery_expr.traverse_leaf_func(leaf_func="get_constraint_tree")
                assert isinstance(operator_tree_expr, ArithmeticExpression)
            else:
                raise Exception
            gc_constraint_trees.append(operator_tree_expr)
            gc_ranges.append((np.float64(gc_range_a), np.float64(gc_range_b)))
        self.coalesced_gcs = get_coalesced_global_constraints(gc_constraint_trees, gc_ranges)
        self.uncoalesced_gcs = get_uncoalesced_global_constraints(self.coalesced_gcs)
        self.coalesced_bcs = get_coalesced_base_constraints(self.bc_query)
        self.uncoalesced_bcs = get_uncoalesced_base_constraints(self.coalesced_bcs)


    def __str__(self):
        return (
            "/-------------------------------------------- PaQL Query ---------------------------------------------\\\n"
            "|  PaQL query:\n"
            "|     " + str(self._paql_query_str) + "\n"
            "|  Base SQL query:\n"
            "|     " + str(self.bc_query) + "\n"
            "|  Global SQL queries:\n"
            "|     " + ("|     ".join([ str(q) + "\n" for q in self.gc_queries ]) if self.gc_queries else "None\n") + ""
            "|  Glogal constraint ranges:\n"
            "|     " + ("|     ".join([ str(q) + "\n" for q in self.gc_ranges ]) if self.gc_ranges else "None\n") + ""
            "|  Optimization objective:\n"
            "|     " + (str(self.objective) if self.objective else "None") + "\n"
            "\-----------------------------------------------------------------------------------------------------/"
        )


    def get_paql_str(self, redo=False, recompute_gcs=True, coalesced=False):
        if redo or self._paql_query_str is None or self._paql_query_str_stale:

            if recompute_gcs:
                self.coalesced_gcs = get_coalesced_global_constraints(self.gc_queries, self.gc_ranges)
                self.uncoalesced_gcs = get_uncoalesced_global_constraints(self.coalesced_gcs)
                self.coalesced_bcs = get_coalesced_base_constraints(self.bc_query)
                self.uncoalesced_bcs = get_uncoalesced_base_constraints(self.coalesced_bcs)

            if self.rel_namespace is None:
                # raise Exception("rel_namespace is None")
                # return ""
                self.rel_namespace = { "R": self.table_name }

            bcs_str = []
            gcs_str = []
            obj_str = None

            if not coalesced:
                if len(self.uncoalesced_bcs) > 0:
                    for attr, op, n in self.uncoalesced_bcs:
                        bcs_str.append("{} {} {}".format(attr, op_to_opstr(op), n))

                if len(self.uncoalesced_gcs) > 0:
                    for (aggr, attr), op, n in self.uncoalesced_gcs:
                        gcs_str.append("{}({}) {} {}".format(aggr, attr, op_to_opstr(op), n))

            else:
                if len(self.coalesced_bcs) > 0:
                    for attr, (lb, ub) in self.coalesced_bcs.iteritems():
                        if float(lb) == -float("inf") and float(ub) == float("inf"):
                            continue
                        elif float(ub) == float("inf"):
                            bcs_str.append("{} {} {}".format(attr, op_to_opstr(operator.ge), lb))
                        elif float(lb) == -float("inf"):
                            bcs_str.append("{} {} {}".format(attr, op_to_opstr(operator.le), ub))
                        elif lb == ub:
                            bcs_str.append("{} {} {}".format(attr, op_to_opstr(operator.eq), ub))
                        else:
                            bcs_str.append("{} BETWEEN {} AND {}".format(attr, lb, ub))

                if len(self.coalesced_gcs) > 0:
                    for (aggr, attr), (lb, ub) in self.coalesced_gcs.iteritems():
                        if aggr.lower() == "count":
                            lb, ub = int(lb), int(ub)

                        uaggr = aggr.upper()

                        if float(lb) == -float("inf") and float(ub) == float("inf"):
                            continue
                        elif float(ub) == float("inf"):
                            gcs_str.append("{}({}) {} {}".format(uaggr, attr, op_to_opstr(operator.ge), lb))
                        elif float(lb) == -float("inf"):
                            gcs_str.append("{}({}) {} {}".format(uaggr, attr, op_to_opstr(operator.le), ub))
                        elif lb == ub:
                            gcs_str.append("{}({}) {} {}".format(uaggr, attr, op_to_opstr(operator.eq), ub))
                        else:
                            gcs_str.append("{}({}) BETWEEN {} AND {}".format(uaggr, attr, lb, ub))

            if self.objective is not None:
                if self.objective["type"] == "maximize":
                    obj_str = "MAXIMIZE "
                elif self.objective["type"] == "minimize":
                    obj_str = "MINIMIZE "
                else:
                    raise
                obj_str += self.objective["func"].get_str()

            self._paql_query_str = \
                "SELECT \n\tPACKAGE({pack}) \n" \
                "FROM \n\t{tables} {bcs}{gcs}{obj};".format(
                pack=", ".join(self.rel_namespace.keys()),
                tables=", ".join("{} {}".format(name, alias) for alias, name in self.rel_namespace.iteritems()),
                bcs="\nWHERE \n\t{} ".format(" AND\n\t".join(bcs_str)) if bcs_str else "",
                gcs="\nSUCH THAT \n\t{} ".format(" AND\n\t".join(gcs_str)) if gcs_str else "",
                obj="\n{}".format(obj_str) if obj_str is not None else "")

            self._paql_query_str_stale = False

        return self._paql_query_str
