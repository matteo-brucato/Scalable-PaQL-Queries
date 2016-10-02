class InfeasiblePackageQuery(Exception):
	def __str__(self):
		return "Package query is infeasible."


class Interrupted(Exception):
	def __str__(self):
		return "Search has been manually or automatically interrupted."


class SubprocessKilled(Exception):
	pass


class TimeLimitElapsed(Exception):
	def __init__(self, time_limit=None, *args, **kwargs):
		super(TimeLimitElapsed, self).__init__(*args, **kwargs)
		self.time_limit = time_limit

	def __str__(self):
		return "Search time limit ({}) has passed.".format(
			" sec.".format(self.time_limit) if self.time_limit is not None else "unknown"
		)



class SQLValueOutOfRange(Exception):
	def __str__(self):
		return "DBMS raised value-out-of-range error during SQL query."



class PowersetTableTooBig(Exception):
	def __str__(self):
		return "The powerset table would be too big to be materialized."



class LPOverflowError(Exception):
	def __str__(self):
		return "Linear programming solver raised overflow error."

