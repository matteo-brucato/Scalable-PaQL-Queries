from abc import ABCMeta, abstractmethod

from src.data_model.tuple import Tuple



class Package(object):
	"""
	A package is simply *any* collection of tuples.
	It may be "infeasible" (or "invalid"): it may violate one or more of the base and/or global constraints.
	"""
	__metaclass__ = ABCMeta


	def __init__(self, search=None, table_name=None):
		"""
		:param search: The Search object which is using this candidate pacakge.
		"""
		self.search = search
		self.table_name = table_name

		if self.search is not None:
			assert self.table_name is None
			self.table_name = self.search.package_table.table_name

		# else:
		# 	self.table_name = None


	def convert_to(self, PackageClass):
		"""
		:type PackageClass: classobj
		"""
		assert issubclass(PackageClass, Package)
		assert self.__class__ != PackageClass
		tuples = [
			Tuple(attrs=None, record=t)
			for t in self.iter_tuples()
		]
		return PackageClass(search=self.search, tuples=tuples)


	@abstractmethod
	def materialize(self, *args, **kwargs):
		raise NotImplemented


	@abstractmethod
	def delete(self, *args, **kwargs):
		raise NotImplemented


	@abstractmethod
	def drop(self, *args, **kwargs):
		raise NotImplemented


	@abstractmethod
	def iter_tuples(self, *args, **kwargs):
		raise NotImplemented


	@staticmethod
	@abstractmethod
	def generate_random_candidate(search):
		"""
		Generates random candidate in the plausibly valid paql_eval space (considering pruning if it is on).
		"""
		raise NotImplemented


	@abstractmethod
	def __len__(self):
		raise NotImplementedError
