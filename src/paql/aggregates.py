from abc import ABCMeta



class Aggr(object):
	__metaclass__ = ABCMeta


	def __init__(self, args):
		self.aggr_name = None
		self.args = tuple(args)


	def __str__(self):
		return "{}({})".format(
			self.aggr_name.upper(),
			",".join(self.args))


	def __hash__(self):
		return hash(str(self))


	def __eq__(self, other):
		return str(self) == str(other)


	def get_sql(self):
		return str(self)


	def get_aggr_attr(self):
		return (self.aggr_name,) + self.args


	def is_simple_linear(self):
		"""
		Returns whether this is a simple linear aggregate (such as COUNT(*) and SUM(protein))
		"""
		return False


	def is_linear(self):
		return False




class SingleArgAggr(Aggr):
	__metaclass__ = ABCMeta


	@property
	def arg(self):
		return self.args[0]


	def __init__(self, args):
		super(SingleArgAggr, self).__init__(args)
		assert len(args) == 1




class SumAggr(SingleArgAggr):
	def __init__(self, args):
		super(SumAggr, self).__init__(args)
		self.aggr_name = "sum"


	def is_simple_linear(self):
		return True


	def is_linear(self):
		return True


	def get_sql(self):
		return "COALESCE({}, 0)".format(str(self))



class AvgAggr(SingleArgAggr):
	def __init__(self, args):
		super(AvgAggr, self).__init__(args)
		self.aggr_name = "avg"


	def is_simple_linear(self):
		return True


	def is_linear(self):
		return True


	def get_sql(self):
		return "COALESCE({}, 0)".format(str(self))



class CountAggr(SingleArgAggr):
	def __init__(self, args):
		super(CountAggr, self).__init__(args)
		self.aggr_name = "count"


	def is_simple_linear(self):
		return True


	def is_linear(self):
		return True



class MaxAggr(SingleArgAggr):
	def __init__(self, args):
		super(MaxAggr, self).__init__(args)
		self.aggr_name = "max"



class MinAggr(SingleArgAggr):
	def __init__(self, args):
		super(MinAggr, self).__init__(args)
		self.aggr_name = "min"
