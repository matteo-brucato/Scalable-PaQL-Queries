from abc import ABCMeta

from src.paql.aggregates import Aggr
from src.paql.expression_trees.expression_trees import SQLQueryExpression
from src.utils.log import log



class ObjectiveSense(object):
	__metaclass__ = ABCMeta



class ObjectiveSenseMAX(ObjectiveSense):
	def __str__(self):
		return "maximize"



class ObjectiveSenseMIN(ObjectiveSense):
	def __str__(self):
		return "minimize"



class PackageQueryObjective(object):
	def __init__(self, sqlquery_expr, sense):
		assert isinstance(sqlquery_expr, SQLQueryExpression)
		assert isinstance(sense, ObjectiveSense)
		self.SQLQuery_expr = sqlquery_expr
		self.sense = sense


	def is_simple_linear(self):
		sql_queries = list(self.SQLQuery_expr.iter_leaves())
		return len(sql_queries) == 1 and sql_queries[0].get_constraint_tree().is_simple_linear()


	def iter_aggregates(self):
		op_tree_expr = self.SQLQuery_expr.traverse_leaf_func("get_constraint_tree")
		for op_tree in op_tree_expr.iter_leaves():
			for op_tree_leaf in op_tree.iter_leaves():
				if isinstance(op_tree_leaf, Aggr):
					yield op_tree_leaf


	def get_aggregate(self):
		log("Getting objective aggregate...")
		aggregates = list(self.iter_aggregates())
		assert len(aggregates) == 1
		return aggregates[0]


	def get_attributes(self):
		attrs = set()
		for aggr in self.iter_aggregates():
			attrs.update(aggr.args)

		return attrs - {"*"}
