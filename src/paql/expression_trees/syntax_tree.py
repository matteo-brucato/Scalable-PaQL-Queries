from src.paql.aggregates import *
from src.paql.expression_trees.expression_trees import *
from src.paql.expression_trees.operator_tree import *
from src.utils.utils import opstr_to_op




class Expression(object):
	"""
	A class that transforms a hierarchy of python objects into a tree of Expr_Node objects.

	Note: The type of the objects returned by this class is of base class Expr_Node, having
	      a different subclass for each type of tree node (binary operations, leaves, etc.).
	      This class doesn't produce objects of class Expression! This is achieved by defining
	      the method __new__ and returning the corret tree node type.
	"""

	def __new__(cls, obj):
		"""
		Returns an expression tree out of a python hierarchy of python dicts as nodes of the tree,
		and ints, floats, strings, etc. as leaves of the tree.
		"""
		if obj is None:
			return Expr_None()

		elif type(obj) is dict:
			# A NODE of the tree (not a leaf, i.e., not a primitive type like str, int, float)

			assert "NODE_TYPE" in obj

			if obj["NODE_TYPE"] == "BIN_OP":
				assert "left" in obj and "right" in obj
				return Expr_BinopNode(Expression(obj["OP"]), Expression(obj["left"]), Expression(obj["right"]))

			elif obj["NODE_TYPE"] == "UN_OP":
				assert "left" in obj
				return Expr_UnopNode(Expression(obj["OP"]), Expression(obj["left"]))

			elif obj["NODE_TYPE"] == "func":
				assert "func_name" in obj and "func_args" in obj
				func_name = obj["func_name"]
				func_args = obj["func_args"]
				return Expr_FuncNode(Expression(func_name), [Expression(arg) for arg in func_args])

			elif obj["NODE_TYPE"] == "COL_REF":
				assert "attr_name" in obj
				attr_name = obj["attr_name"]
				table_name = obj.get("table_name", None)
				return Expr_ColrefNode(Expression(attr_name), Expression(table_name))

			elif obj["NODE_TYPE"] == "SUB_QUERY":
				assert "CONTENT" in obj
				content = obj["CONTENT"]
				# In sub-queries we only allow one select item (not a list), so it's just one expression
				select_expr = Expression(content["SELECT"])
				# In sub-queries we only allow one from relation (not a list), so it's just one from object
				from_e = content["FROM"]
				from_list = RelationReference(rel_name=from_e["REL_NAME"], rel_alias=from_e.get("REL_ALIAS"))
				where_exp = Expression(content["WHERE"])
				group_by_exp = Expression(content["GROUP BY"])
				return Expr_SubQueryNode(select_expr, from_list, where_exp, group_by_exp)

			elif obj["NODE_TYPE"] == "ALL":
				assert "CONTENT" in obj
				content = Expression(obj["CONTENT"])
				return Expr_AllNode(content)

			else:
				raise Exception("NODE_TYPE not recognized: {}".format(obj["NODE_TYPE"]))

		else:
			return Expr_Leaf(obj)



class Expr_Node(object):
	__metaclass__ = ABCMeta


	@abstractmethod
	def __repr__(self):
		raise Exception("This method is abstract.")


	@abstractmethod
	def get_str(self, parentheses=True):
		raise Exception("This method is abstract.")


	@abstractmethod
	def is_conjunctive(self):
		raise Exception("This method is abstract.")


	@abstractmethod
	def get_tuple_level_expression(self):
		raise Exception("This method is abstract.")


	@abstractmethod
	def get_aggr_arithmetic_expression(self):
		raise Exception("This method is abstract.")


	@abstractmethod
	def get_sql_arithmetic_expression(self):
		raise Exception("This method is abstract.")


	@abstractmethod
	def get_ANDed_gc_list(self, gc_list=None):
		"""
		Return list of global constraints for expression with aggregations.
		This ugly function is assuming that the SUCH THAT clause is an ANDed (i.e. conunctive) expression.
		It produces a list of ANDed global contraints. In the future,
		this will be removed and replaced with something more general,
		or kept for conjuntive queries only.

		:rtype : list of global constraints formatted as follows: each element is a
		tuple (x, a, b) encoding the constraint a <= x <= b, where x is a SQL query
		of the form SELECT func(args,...) FROM core_table, and a and b are numbers.
		"""
		# TODO: This function needs to be either eliminated or modified when support for generic expressions (with OR
		# and NOT) will be provided. In fact, in that case we can't return a "list", but we need to maintain the tree
		# structure of the expression.
		raise Exception("This method is abstract.")



class Expr_AllNode(Expr_Node):
	def __init__(self, content):
		# For now, make sure that content of ALL expressions are sub-queries only:
		assert isinstance(content, Expr_SubQueryNode)
		self.content = content

	def __repr__(self):
		return "Expr_AllNode[{}]".format(repr(self.content))


	def get_str(self, parentheses=True):
		return self.content.get_str()


	def is_conjunctive(self):
		return True


	def get_tuple_level_expression(self):
		raise Exception


	def get_aggr_arithmetic_expression(self):
		raise Exception


	def get_sql_arithmetic_expression(self):
		raise NotImplementedError()


	def get_ANDed_gc_list(self, gc_list=None):
		return self.content.get_ANDed_gc_list(gc_list)



class Expr_SubQueryNode(Expr_Node):
	"""
	Sub-query node.
	"""

	def __init__(self, select_expr, rel_ref, where_exp, group_by_exp):
		assert isinstance(select_expr, Expr_Node)
		assert isinstance(rel_ref, RelationReference)
		assert isinstance(where_exp, Expr_Node)
		assert isinstance(group_by_exp, Expr_Node)
		self.sub_query = SQLQuery(
			select_list=[ select_expr ],
			from_list=[ rel_ref ],
			where_expr=where_exp,
			group_by_expr=group_by_exp)


	def __repr__(self):
		return "Expr_SubQueryNode[{}]".format(repr(self.sub_query))


	def get_str(self, parentheses=True):
		return self.sub_query.get_str()


	def is_conjunctive(self):
		return True


	def get_tuple_level_expression(self):
		raise Exception


	def get_aggr_arithmetic_expression(self):
		raise Exception("TODO: Implement this.")
		assert len(self.sub_query.select_list) == 1
		return self.sub_query.select_list[0].get_aggr_arithmetic_expression()


	def get_sql_arithmetic_expression(self):
		"""
		Returns an arithmetic expression tree where the leaves are SQLQuery objects.
		"""
		return ArithmeticExpressionLEAF(self.sub_query)
		# return self.sub_query


	def get_ANDed_gc_list(self, gc_list=None):
		return self.get_sql_arithmetic_expression()



class Expr_BinopNode(Expr_Node):
	"""
	Binary operator node. For instance, "A AND B", "A OR B", "A+B", "A-B", "A*B", etc.
	"""

	def __init__(self, op, left, right):
		assert isinstance(op, Expr_Leaf)
		assert isinstance(left, Expr_Node)
		assert isinstance(right, Expr_Node)
		self.op = op
		self.left = left
		self.right = right


	def __repr__(self):
		return "Expr_BinopNode[{},{}]".format(repr(self.left), repr(self.right))


	def get_str(self, parentheses=True):
		if not parentheses:
			st = "{} {} {}"
		else:
			st = "({}) {} ({})"
		return st.format(
			self.left.get_str(),
			self.op.get_str(),
			self.right.get_str()

		)

	def is_conjunctive(self):
		return self.op.val != "not" and self.op.val != "or" \
		       and self.left.is_conjunctive() and self.right.is_conjunctive()


	def get_tuple_level_expression(self):
		if self.op.val == "-" or self.op.val == "/" or self.op.val == "*" or self.op.val == "+":
			left = self.left.get_tuple_level_expression()
			right = self.right.get_tuple_level_expression()
			return ArithmeticExpressionBINOP(opstr_to_op(self.op.val), left, right)
		elif self.op.val == "and" or self.op.val == "or":
			left = self.left.get_tuple_level_expression()
			right = self.right.get_tuple_level_expression()
			return LogicalExpressionBINOP(opstr_to_op(self.op.val), left, right)
		elif self.op.val == "<=" or self.op.val == ">=" or self.op.val == "=":
			left = self.left.get_tuple_level_expression()
			right = self.right.get_tuple_level_expression()
			return ComparisonExpressionBINOP(opstr_to_op(self.op.val), left, right)
		else:
			raise Exception("Exception: math op not supported: {}".format(self.op))


	def get_aggr_arithmetic_expression(self):
		if self.op.val == "-" or self.op.val == "/" or self.op.val == "*" or self.op.val == "+":
			left = self.left.get_aggr_arithmetic_expression()
			right = self.right.get_aggr_arithmetic_expression()
			return ArithmeticExpressionBINOP(opstr_to_op(self.op.val), left, right)
		else:
			raise Exception("Exception: math op not supported: {}".format(self.op))


	def get_sql_arithmetic_expression(self):
		if self.op.val == "-" or self.op.val == "/" or self.op.val == "*" or self.op.val == "+":
			left = self.left.get_sql_arithmetic_expression()
			right = self.right.get_sql_arithmetic_expression()
			return ArithmeticExpressionBINOP(opstr_to_op(self.op.val), left, right)
		else:
			raise Exception("Exception: math op not supported: {}".format(self.op))


	def get_ANDed_gc_list(self, gc_list=None):
		assert isinstance(self.op, Expr_Leaf)

		if gc_list is None:
			gc_list = []

		# LOGICAL OPERATOR (only support AND for now)
		if self.op.val == "and":
			self.left.get_ANDed_gc_list(gc_list)
			self.right.get_ANDed_gc_list(gc_list)

		# CONSTRAINT OPERATOR
		# TODO: Support other binary operators. Notice that < and > actually require NOT to be implemented as well.
		elif self.op.val == "=" or self.op.val == "<=" or self.op.val == ">=":
			# SQLQueryExpression <=/>=/= val   or   val <=/>=/= SQLQueryExpression
			left = self.left.get_ANDed_gc_list()
			right = self.right.get_ANDed_gc_list()

			# left must be a SQLQueryExpression (expression of SQLQuery's)
			# if isinstance(right, SQLQueryExpression):  # or isinstance(right, SQLQuery):
			# 	left, right = right, left
			# assert isinstance(left, SQLQueryExpression)  # or isinstance(left, SQLQuery)

			# NOTE: Not supporting SQLQueryExpression <=/>=/= SQLQueryExpression

			# We assume that only one of the two sides is a SQLQueryExpression
			if isinstance(left, SQLQueryExpression) and not isinstance(right, SQLQueryExpression):
				sql_expr = left
				rhs = right
				op = self.op.val
			elif isinstance(right, SQLQueryExpression) and not isinstance(left, SQLQueryExpression):
				sql_expr = right
				rhs = left
				if self.op.val == "<=":
					op = ">="
				elif self.op.val == ">=":
					op = "<="
				else:
					op = self.op.val
			else:
				raise Exception

			# TODO: Support other binary operators.
			if op == "=":
				gc_list.append((sql_expr, rhs, rhs))

			elif op == "<=":
				gc_list.append((sql_expr, -float("inf"), rhs))

			elif op == ">=":
				gc_list.append((sql_expr, rhs, float("inf")))

			else:
				raise Exception("Exception: constraint op not supported: {}".format(self.op))

		# ARITHMETIC OPERATOR
		else:
			return self.get_sql_arithmetic_expression()

		return gc_list



class Expr_UnopNode(Expr_Node):
	"""
	Unary operator node. For example "NOT A", "-A", "+A", etc.
	"""

	def __init__(self, op, left):
		assert isinstance(op, Expr_Leaf) and isinstance(left, Expr_Node)
		self.op = op
		self.left = left


	def __repr__(self):
		return "Expr_UnopNode[{},{}]".format(repr(self.left))


	def get_str(self, parentheses=True):
		return "{} {}".format(
			self.op.get_str(),
			self.left.get_str()
		)


	def is_conjunctive(self):
		return self.left.is_conjunctive()


	def get_tuple_level_expression(self):
		if self.op.val == "-" or self.op.val == "+":
			left = self.left.get_aggr_arithmetic_expression()
			return ArithmeticExpressionUNOP(opstr_to_op(self.op.val), left)
		elif self.op.val == "not":
			left = self.left.get_aggr_arithmetic_expression()
			return LogicalExpressionUNOP(opstr_to_op(self.op.val), left)
		else:
			raise Exception("Exception: math unop not supported: {}".format(self.op))


	def get_aggr_arithmetic_expression(self):
		if self.op.val == "-" or self.op.val == "+":
			left = self.left.get_aggr_arithmetic_expression()
			return ArithmeticExpressionUNOP(opstr_to_op(self.op.val), left)
		else:
			raise Exception("Exception: math unop not supported: {}".format(self.op))


	def get_sql_arithmetic_expression(self):
		if self.op.val == "-" or self.op.val == "+":
			left = self.left.get_sql_arithmetic_expression()
			return ArithmeticExpressionUNOP(opstr_to_op(self.op.val), left)
		else:
			raise Exception("Exception: math unop not supported: {}".format(self.op))


	def get_ANDed_gc_list(self, gc_list=None):
		raise Exception("Unary operator '{}' not supported for function get_ANDed_gc_list()".format(self.op))



class Expr_FuncNode(Expr_Node):
	"""
	Function node. For instance, "SUM(fat)", "COUNT(*)", "f(x, y)", etc.
	"""

	def __init__(self, func_name, func_args):
		assert isinstance(func_name, Expr_Leaf) and isinstance(func_args, list)
		self.func_name = func_name
		self.func_args = func_args


	def __repr__(self):
		return "Expr_FuncNode[{}({})]".format(repr(self.func_name), ",".join(repr(arg) for arg in self.func_args))


	def get_str(self, parentheses=True):
		return "{}({})".format(
			self.func_name.get_str(),
			",".join([arg.get_str() for arg in self.func_args]),
		)


	def is_conjunctive(self):
		return True


	def get_tuple_level_expression(self):
		raise Exception()


	def get_aggr_arithmetic_expression(self):
		# raise Exception("Reactivate this code. (Should you return a SQLQuery instead?)")
		if len(self.func_args) == 1:
			args = [arg.get_str() for arg in self.func_args]

			# NOTE: For now, supporting aggregate functions with only one argument
			if self.func_name.get_str().lower() == "sum":
				aggregate = SumAggr

			elif self.func_name.get_str().lower() == "count":
				aggregate = CountAggr

			elif self.func_name.get_str().lower() == "avg":
				aggregate = AvgAggr

			elif self.func_name.get_str().lower() == "min":
				aggregate = MinAggr

			elif self.func_name.get_str().lower() == "max":
				aggregate = MaxAggr

			else:
				raise Exception("Not supported")
		else:
			raise Exception("Not supported")

		return ArithmeticExpressionLEAF(aggregate(args))


	def get_sql_arithmetic_expression(self):
		"""
		Returns an arithmetic expression tree where the leaves are SQLQuery objects.
		"""
		return ArithmeticExpressionLEAF(SQLQuery(
			select_list=[self],
			from_list=["memory_representations"],
			where_expr=Expr_None(),
			group_by_expr=Expr_None()))


	def get_ANDed_gc_list(self, gc_list=None):
		return self.get_sql_arithmetic_expression()



class Expr_ColrefNode(Expr_Node):
	"""
	Column reference node. For instance, "fat", "R.fat", "P.fat", "recipe.fat", etc.
	"""

	def __init__(self, attr_name, table_name):
		assert isinstance(attr_name, Expr_Leaf) and (
		isinstance(table_name, Expr_Leaf) or isinstance(table_name, Expr_None))
		self.attr_name = attr_name
		self.table_name = table_name


	def __repr__(self):
		return "Expr_ColrefNode[{}.{}]".format(repr(self.table_name), repr(self.attr_name))


	def get_str(self, parentheses=True):
		table_name_str = self.table_name.get_str()
		return "{}{}".format(
			table_name_str + "." if table_name_str else "",
			self.attr_name.get_str(),
		)


	def is_conjunctive(self):
		return True


	def get_tuple_level_expression(self):
		return ExpressionLEAF(self.get_str())


	def get_aggr_arithmetic_expression(self):
		return self.get_str()


	def get_sql_arithmetic_expression(self):
		return self.get_str()


	def get_ANDed_gc_list(self, gc_list=None):
		return self.get_str()



class Expr_Leaf(Expr_Node):
	"""
	Leaf node. A string, a number, or "infinity".
	"""

	def __init__(self, val):
		if isinstance(val, basestring):
			if val == "infinity":
				self.val = float("inf")
			else:
				self.val = val.lower()
		elif type(val) is float or type(val) is int:
			self.val = val
		else:
			raise AssertionError("Type of Expression leaf should be string, float, or int.")


	def __repr__(self):
		return "Expr_Leaf[{}]".format(self.val)


	def get_str(self, parentheses=True):
		if isinstance(self.val, basestring):
			return self.val

		elif type(self.val) is float:
			return "{:.10f}".format(self.val)

		elif type(self.val) is int:
			return "{:d}".format(self.val)

		elif type(self.val) is long:
			return "{:ld}".format(self.val)

		else:
			raise Exception("Type of Expr_Leaf val unsupported: {}".format(type(self.val)))


	def is_conjunctive(self):
		return True


	def get_tuple_level_expression(self):
		return ExpressionLEAF(self.val)


	def get_aggr_arithmetic_expression(self):
		return ArithmeticExpressionLEAF(leaf=self.get_str())


	def get_sql_arithmetic_expression(self):
		return self.get_str()


	def get_ANDed_gc_list(self, gc_list=None):
		if isinstance(self.val, basestring):
			raise Exception()
			# return "(SELECT {} FROM memory_representations)".format(self.val)

		elif type(self.val) is float:
			return self.val

		elif type(self.val) is int:
			return self.val

		else:
			raise Exception("Unsupported type for self.val: {}".format(type(self.val)))



class Expr_None(Expr_Node):
	"""
	No-expression node. It corresponds to the absence of an expression.
	"""

	def __init__(self):
		pass


	def __repr__(self):
		return "Expr_None"


	def get_str(self, parentheses=True):
		return ""


	def is_conjunctive(self):
		return True


	def get_tuple_level_expression(self):
		return ExpressionLEAF(self.get_str())


	def get_aggr_arithmetic_expression(self):
		return self.get_str()


	def get_sql_arithmetic_expression(self):
		return self.get_str()


	def get_ANDed_gc_list(self, gc_list=None):
		if gc_list is None:
			gc_list = []

		return gc_list



class RelationReference:
	def __init__(self, rel_name, rel_alias=None):
		self.rel_name = rel_name
		self.rel_alias = rel_alias
		if self.rel_alias is None:
			self.rel_alias = self.rel_name

	def __str__(self):
		return self.rel_name



class SQLQuery:
	def __init__(self, select_list, from_list, where_expr, group_by_expr):
		assert isinstance(where_expr, Expr_Node)
		assert isinstance(group_by_expr, Expr_Node)
		self.select_list = select_list
		self.from_list = from_list
		self.where_expr = where_expr
		self.group_by_expr = group_by_expr


	def __hash__(self):
		return hash(self.get_str())


	def __eq__(self, other):
		return self.get_str() == other.get_str()


	def __str__(self):
		return self.get_str()


	def __repr__(self):
		return self.get_str()


	def __iter__(self):
		yield self


	# def map(self, func):
	# 	yield func(self)


	def get_str(self):
		where_clause = self.where_expr.get_str()
		group_by_clause = self.group_by_expr.get_str()
		return "SELECT {} FROM {}{where_clause}{group_by_clause}".format(
			", ".join(s.get_str() for s in self.select_list),
			", ".join(str(f) for f in self.from_list),
			where_clause=(" WHERE " + where_clause) if where_clause != "" else "",
			group_by_clause=(" GROUP BY " + group_by_clause) if group_by_clause != "" else "",
		)


	def get_constraint_tree(self):
		"""
		Returns a tree corresponding to this SQL query (like a sort of query plan).
		This will include:
		- Arithmetic expression of aggregate (Aggr) objects;
		- Other operators such as GROUP BY, or SELECTION.
		- The order matters: this structure will be also used as a sort of "constraint plan" for global constraints.
		NOTE: For now, it assumes that this query is a single-attribute, single-aggregate query. More features
		will be added later.
		"""
		assert len(self.from_list) == 1

		if len(self.select_list) == 1:
			# No WHERE and no GROUP BY
			if isinstance(self.where_expr, Expr_None) and isinstance(self.group_by_expr, Expr_None):
				return AggregateOperatorLeaf(
					aggr_arithmetic_expr=self.select_list[0].get_aggr_arithmetic_expression())

			# With WHERE clause
			elif not isinstance(self.where_expr, Expr_None) and isinstance(self.group_by_expr, Expr_None):
				next_node = AggregateOperatorLeaf(
					aggr_arithmetic_expr=self.select_list[0].get_aggr_arithmetic_expression())
				return SelectionOperatorNode(
					selection_expr=self.where_expr.get_tuple_level_expression(),
					next_node=next_node)

			# With GROUP BY clause
			elif isinstance(self.where_expr, Expr_None) and not isinstance(self.group_by_expr, Expr_None):
				next_node = AggregateOperatorLeaf(
					aggr_arithmetic_expr=self.select_list[0].get_aggr_arithmetic_expression())
				return GroupByOperatorNode(
					group_by_expr=self.group_by_expr.get_tuple_level_expression(),
					next_node=next_node)

		raise Exception("Not supported: {}".format(self))



