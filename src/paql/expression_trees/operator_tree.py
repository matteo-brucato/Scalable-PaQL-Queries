from abc import ABCMeta, abstractmethod

from src.paql.expression_trees.expression_trees import ArithmeticExpression



class OperatorNode(object):
	__metaclass__ = ABCMeta


	@abstractmethod
	def __iter__(self):
		raise NotImplementedError


	def __hash__(self):
		return hash(str(self))


	def __eq__(self, other):
		return str(self) == str(other)


	def is_leaf(self):
		return False


	def iter_leaves(self):
		for x in self:
			if x.is_leaf():
				# for l in x.iter_leaves():
				yield x.leaf

	def is_simple_linear(self):
		return False



class AggregateOperatorLeaf(OperatorNode):
	@property
	def leaf(self):
		return self.aggr_arithmetic_expr


	def __init__(self, aggr_arithmetic_expr):
		assert isinstance(aggr_arithmetic_expr, ArithmeticExpression)
		self.aggr_arithmetic_expr = aggr_arithmetic_expr


	def __iter__(self):
		yield self
		for l in self.aggr_arithmetic_expr:
			yield l


	# def iter_leaves(self):
	# 	for l in self.aggr_arithmetic_expr.iter_leaves():
	# 		yield l


	def is_leaf(self):
		return True


	def is_simple_linear(self):
		return self.aggr_arithmetic_expr.is_simple_linear()


	def __str__(self):
		return str(self.aggr_arithmetic_expr)



class GroupByOperatorNode(OperatorNode):
	def __init__(self, group_by_expr, next_node):
		# isinstance(group_by_expr, Expression)
		assert isinstance(next_node, OperatorNode)
		self.group_by_expr = group_by_expr
		self.next = next_node


	def __iter__(self):
		yield self
		for l in self.group_by_expr:
			yield l
		for l in self.next:
			yield l


	def __str__(self):
		return str(self.next) + " GROUP BY " + str(self.group_by_expr)



class SelectionOperatorNode(OperatorNode):
	def __init__(self, selection_expr, next_node):
		# isinstance(selection_expr, Expression)
		assert isinstance(next_node, OperatorNode)
		self.selection_expr = selection_expr
		self.next = next_node


	def __iter__(self):
		yield self
		for l in self.selection_expr:
			yield l
		for l in self.next:
			yield l


	def __str__(self):
		return str(self.next) + " WHERE " + str(self.selection_expr)
