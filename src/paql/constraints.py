import operator
from abc import ABCMeta
from itertools import izip

from src.paql.aggregates import Aggr, SingleArgAggr
from src.paql.expression_trees.expression_trees import ArithmeticExpression
from src.utils.utils import op_to_opstr



class GlobalConstraint(object):
	__metaclass__ = ABCMeta

	def __init__(self, expr):
		"""
		expr: An arithmetic expression of Operator Nodes
		"""
		assert isinstance(expr, ArithmeticExpression)
		self.expr = expr


	def is_simple_linear(self):
		return self.expr.is_simple_linear()


	def iter_aggregates(self):
		for tree_node in self.expr:
			if tree_node.is_leaf():
				for expr_node in tree_node.leaf:
					if expr_node.is_leaf() and isinstance(expr_node.leaf, Aggr):
							yield expr_node.leaf


	def get_attributes(self):
		attrs = set()
		for aggr in self.iter_aggregates():
			attrs.update(aggr.args)
		return attrs - {"*"}




class CGlobalConstraint(GlobalConstraint):
	"""
	A Coalesced Global Constraint contains:
	- An arithmetic expression tree of SQLQuery objects
	- A bound: a pair of values indicating an interval where the value of the constraint must reside
	"""
	def __init__(self, expr, lb, ub):
		super(self.__class__, self).__init__(expr)
		assert type(lb) is float or type(lb) is int
		assert type(ub) is float or type(ub) is int
		self.lb = lb
		self.ub = ub


	def __str__(self):
		return " ".join([
			self.expr.leaf.__class__.__name__,
			str(self.expr),
			"BETWEEN",
			str(self.lb),
			"AND",
			str(self.ub)
		])



class UGlobalConstraint(GlobalConstraint):
	"""
	An Uncoalesced Global Constraint contains:
	- An arithmetic expression tree of SQLQuery objects
	- An operator
	- A right-hand value
	"""
	def __init__(self, expr, op, rhs):
		super(self.__class__, self).__init__(expr)
		assert op is operator.le or op is operator.ge or op is operator.eq
		assert type(rhs) is float or type(rhs) is int
		self.op = op
		self.rhs = rhs


	def __str__(self):
		return " ".join([
			str(self.expr),
			op_to_opstr(self.op),
			str(self.rhs)
		])


	def is_simple_linear(self):
		return (self.op is operator.le or self.op is operator.ge or self.op is operator.eq) \
		       and super(self.__class__, self).is_simple_linear()



def convert_value(val):
	if float(val) == float("inf") or float(val) == -float("inf"):
		return val
	floated = float(val)
	inted = int(floated)
	if floated == inted:
		return inted
	return floated



def get_coalesced_base_constraints(bc_query):
	"""
	NOTE: It assumes all clauses are in AND
	Takes an SQL query and extracts all clauses in AND. It returns a dictionary where
	keys are column names (i.e. attributes) and values are pairs (lb, ub) indicating that:
        lb <= attribute <= ub.
	"""
	# TODO: Make sure you don't call this function on a query containing OR's or NOT's in the WHERE clause.
	# FIXME: Use parsed trees of formulas instead of the bc_query and gc_query

	bconstraints = { }

	if ' limit ' in bc_query.lower():
		limit = int(bc_query.lower().split(' limit ')[1])
		bconstraints['limit'] = limit

	if ' where ' in bc_query.lower():
		cs = bc_query.lower().split(' where ')[1].split(' and ')

		for c in cs:
			if c.lower().strip() == "true":
				continue
			elif '<=' in c:
				rel = '<='
				attr, const = c.split('<=')
			elif '>=' in c:
				rel = '>='
				attr, const = c.split('>=')
			elif '=' in c:
				rel = '='
				attr, const = c.split('=')
			else:
				raise Exception("This operator is not supported yet: {}".format(c))

			attr = attr.strip()

			if attr not in bconstraints:
				# bconstraints[attr] = [-sys.maxint, sys.maxint]
				bconstraints[attr] = [ -float("inf"), float("inf") ]
			if rel == '<=':
				bconstraints[attr][1] = convert_value(const)
			elif rel == '>=':
				bconstraints[attr][0] = convert_value(const)
			elif rel == '=':
				bconstraints[attr][0] = bconstraints[attr][1] = convert_value(const)
			else:
				raise

		for attr, bounds in bconstraints.iteritems():
			assert bconstraints[attr] == sorted(bconstraints[attr])

	return bconstraints



def get_uncoalesced_base_constraints(coalesced_bcs):
	"""
	NOTE: It assumes all clauses are in AND
	Takes an SQL query and extracts all clauses in AND. It returns a list of constraints of the form:
		(attr, op, n), where op is an operation, e.g <=, >=, =, etc.
	"""
	# TODO: Make sure you don't call this function on a query containing OR's or NOT's in the WHERE clause.
	# FIXME: Use parsed trees of formulas instead of the bc_query and gc_query

	bconstraints = []

	# TODO: I'm assuming only <=, =, and >= are present in the query. You'll need to support more operators
	# bcs = get_coalesced_base_constraints(bc_query)
	for attr, (lb, ub) in coalesced_bcs.iteritems():
		# if float(lb) == float(-sys.maxint) and float(ub) == float(sys.maxint):
		if float(lb) == -float("inf") and float(ub) == float("inf"):
			# This is the vacuous constraint, always true
			continue
		# elif float(ub) == float(sys.maxint):
		elif float(ub) == float("inf"):
			bconstraints.append(( attr.strip(), operator.ge, lb ))
		# elif float(lb) == float(-sys.maxint):
		elif float(lb) == -float("inf"):
			bconstraints.append(( attr.strip(), operator.le, ub ))
		elif lb == ub:
			# bconstraints.append(( attr.strip(), operator.eq, lb ))
			# NOTE: I'm not using = for this case just to keep <= and >= all well separated
			bconstraints.append(( attr.strip(), operator.ge, lb ))
			bconstraints.append(( attr.strip(), operator.le, ub ))
		else:
			bconstraints.append(( attr.strip(), operator.ge, lb ))
			bconstraints.append(( attr.strip(), operator.le, ub ))

	return bconstraints



def get_uncoalesced_global_constraints(coalesced_gcs):
	"""
	NOTE: It assumes all clauses are in AND
	Takes an SQL query and extracts all clauses in AND. It returns a list of constraints of the form:
		((aggr, attr), op, n), where op is an operation, e.g <=, >=, =, etc.
	"""
	# TODO: Make sure you don't call this function on a query containing OR's or NOT's in the WHERE clause.
	# FIXME: Use parsed trees of formulas instead of the bc_query and gc_query

	gconstraints = []

	# TODO: I'm assuming only <=, =, and >= are present in the query. You'll need to support more operators
	# gcs = get_coalesced_global_constraints(gc_queries, gc_ranges)
	# for (aggr, attr), (lb, ub) in coalesced_gcs.iteritems():
	# for aggregate, (lb, ub) in coalesced_gcs.iteritems():
	for ugc in coalesced_gcs:
		assert isinstance(ugc, CGlobalConstraint)

		if float(ugc.lb) == -float("inf") and float(ugc.ub) == float("inf"):
			# This is the vacuous constraint, always true
			continue
		elif float(ugc.ub) == float("inf"):
			gconstraints.append(UGlobalConstraint(ugc.expr, operator.ge, ugc.lb))
		elif float(ugc.lb) == -float("inf"):
			gconstraints.append(UGlobalConstraint(ugc.expr, operator.le, ugc.ub))
		elif ugc.lb == ugc.ub:
			# NOTE: I'm not using = for this case just to keep <= and >= all well separated
			gconstraints.append(UGlobalConstraint(ugc.expr, operator.ge, ugc.lb))
			gconstraints.append(UGlobalConstraint(ugc.expr, operator.le, ugc.ub))
		else:
			gconstraints.append(UGlobalConstraint(ugc.expr, operator.ge, ugc.lb))
			gconstraints.append(UGlobalConstraint(ugc.expr, operator.le, ugc.ub))

	return gconstraints



def get_coalesced_global_constraints(exprs, ranges):
	"""
	NOTE: It assumes all clauses are in AND
	Takes the SQL queries and corresponding ranges for global constraints and coalesces them.
	It returns a dictionary where keys are aggregation name and column names (i.e. attributes), and values are pairs
	(lb, ub) indicating that:
        lb <= <global-constraint-expression> <= ub.
	"""
	# TODO: Make sure you don't call this function on a query containing OR's or NOT's in the SUCH THAT clause.
	c_gcs = {}

	for expr, crange in izip(exprs, ranges):
		if expr not in c_gcs:
			c_gcs[expr] = CGlobalConstraint(
				expr,
				convert_value(-float("inf")),
				convert_value(float("inf")))

		c_gcs[expr].lb = convert_value(max(c_gcs[expr].lb, crange[0]))
		c_gcs[expr].ub = convert_value(min(c_gcs[expr].ub, crange[1]))

	return c_gcs.values()
