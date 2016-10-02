import operator
from abc import abstractmethod, ABCMeta

from src.utils.utils import op_to_opstr



class ArithmeticExpression(object):
	"""
	A tree expression where the nodes are arithmetic operators and the leaves are SQLQuery objects.
	"""
	__metaclass__ = ABCMeta

	@staticmethod
	def _check_op(op):
		assert op is operator.add or \
		       op is operator.sub or \
		       op is operator.div or \
		       op is operator.mul


	@abstractmethod
	def map(self, func):
		raise NotImplementedError


	@abstractmethod
	def traverse_leaf_func(self, leaf_func):
		raise NotImplementedError


	@abstractmethod
	def __iter__(self):
		raise NotImplementedError


	def is_leaf(self):
		return False


	def iter_leaves(self):
		for x in self:
			# if isinstance(x, ArithmeticExpression) and x.is_leaf():
			if x.is_leaf():
				yield x.leaf

	@abstractmethod
	def is_simple_linear(self):
		return NotImplementedError


# Expression trees of SQLQuery's are arithmetic expression trees where the leaves are SQLQuery objects.
SQLQueryExpression = ArithmeticExpression


class ArithmeticExpressionBINOP(ArithmeticExpression):

	def __init__(self, op, left, right):
		self._check_op(op)
		# assert isinstance(left, ArithmeticExpression)
		# assert isinstance(right, ArithmeticExpression)
		self.op = op
		self.left = left
		self.right = right


	def __hash__(self):
		return hash((self.op, self.left.__hash__(), self.right.__hash__()))


	def __eq__(self, other):
		return (self.op, self.left, self.right) == (other.op, other.left, other.right)
		# return self.op == other.op and self.left == other.left and self.right == other.right


	def __str__(self):
		s = [
			"(", str(self.left), ")",
			op_to_opstr(self.op),
			"(", str(self.right), ")",
		]
		return "".join(s)


	def map(self, func):
		yield func(self)
		for l in self.left:
			yield func(l)
		for r in self.right:
			yield func(r)


	def traverse_leaf_func(self, leaf_func):
		if isinstance(self.left, ArithmeticExpression):
			left_result = self.left.traverse_leaf_func(leaf_func)
		else:
			left_result = getattr(self.left, leaf_func)()
		if isinstance(self.right, ArithmeticExpression):
			right_result = self.right.traverse_leaf_func(leaf_func)
		else:
			right_result = getattr(self.right, leaf_func)()
		return ArithmeticExpressionBINOP(self.op, left_result, right_result)


	def __iter__(self):
		"""
		Implements in-visit traversal.
		"""
		for l in self.left:
			yield l
		yield self
		for r in self.right:
			yield r


	def is_simple_linear(self):
		return self.op == operator.add and self.left.is_simple_linear() and self.right.is_simple_linear()



class ArithmeticExpressionUNOP(ArithmeticExpression):
	def __init__(self, op, left):
		self._check_op(op)
		# assert isinstance(left, ArithmeticExpression)
		self.op = op
		self.left = left


	def __hash__(self):
		return hash((self.op, self.left.__hash__()))


	def __eq__(self, other):
		return (self.op, self.left) == (other.op, other.left)


	def __str__(self):
		s = [
			op_to_opstr(self.op),
			"(", str(self.left), ")",
		]
		return "".join(s)


	def map(self, func):
		yield func(self)
		for l in self.left:
			yield func(l)


	def traverse_leaf_func(self, leaf_func):
		if isinstance(self.left, ArithmeticExpression):
			left_result = self.left.traverse_leaf_func(leaf_func)
		else:
			left_result = getattr(self.left, leaf_func)()
		return ArithmeticExpressionUNOP(self.op, left_result)


	def __iter__(self):
		"""
		Implements in/pre-visit traversal.
		"""
		yield self
		for l in self.left:
			yield l


	def is_simple_linear(self):
		return self.op is operator.pos and self.left.is_simple_linear()



class LogicalExpressionUNOP(ArithmeticExpressionUNOP):
	@staticmethod
	def _check_op(op):
		assert op is operator.not_



class LogicalExpressionBINOP(ArithmeticExpressionBINOP):
	@staticmethod
	def _check_op(op):
		assert op is operator.and_ or op is operator.or_



class ComparisonExpressionBINOP(ArithmeticExpressionBINOP):
	@staticmethod
	def _check_op(op):
		assert op is operator.le or op is operator.ge or op is operator.eq



class ArithmeticExpressionLEAF(ArithmeticExpression):
	def __init__(self, leaf):
		self.leaf = leaf


	def __hash__(self):
		return hash(self.leaf)


	def __eq__(self, other):
		return self.leaf == other.leaf


	def __str__(self):
		return str(self.leaf)


	def __iter__(self):
		yield self


	def is_leaf(self):
		return True


	def map(self, func):
		yield func(self)
		for l in self.leaf:
			yield func(l)


	def traverse_leaf_func(self, leaf_func):
		leaf_result = getattr(self.leaf, leaf_func)()
		# Only create LEAF nodes at the leaf level
		if isinstance(leaf_result, ArithmeticExpression):
			return leaf_result
		else:
			return ArithmeticExpressionLEAF(leaf_result)


	def is_simple_linear(self):
		return self.leaf.is_simple_linear()


ExpressionLEAF = ArithmeticExpressionLEAF
LogicalExpressionLEAF = ArithmeticExpressionLEAF
ComparisonExpressionLEAF = ArithmeticExpressionLEAF

