from abc import abstractmethod, ABCMeta
from src.data_model.core_table import CoreTable
from src.data_model.memory_representation.data_in_memory import InMemoryPackage
from src.data_model.memory_representation.sequence_based import SequenceBasedPackage
from src.paql.package_query import PackageQuery
from src.dbms.db import DBCursor



class PackageStorageModel(object):
	__metaclass__ = ABCMeta


	@abstractmethod
	def __init__(self, db, package_query):
		assert isinstance(db, DBCursor)
		assert isinstance(package_query, PackageQuery)
		self._db = db
		self._package_query = package_query
		self._tablename = None

		self.coretable = CoreTable(self._db, self._package_query)
		self.current_package = None

		# Info:
		# 1. You need to set self.tablename to some unique table name
		# 2. You need to create a view or table in the DB with that table name that will logically store
		#    a candidate package. You know that before reading from that table, users will always call
		#    self.update() first, and pass the actual package that the candidate table is supposed to store.
		# 3. Implement self.update(package) that updates the logical view or table with the current package


	@property
	def table_name(self):
		return self._tablename


	@table_name.setter
	def table_name(self, tablename):
		self._tablename = tablename


	def get_str_list(self):
		return [
			"Package query: {}".format(self._package_query),
			"Table name: {}".format(self.table_name),
			"Core table: {}".format(self.coretable),
			"Current package: {}".format(self.current_package),
		]


	def __str__(self):
		s = [ "Package Data Model (abstract class):" ]
		s.extend(self.get_str_list())
		return "\n".join(s)


	@abstractmethod
	def get_core_table(self):
		raise NotImplemented


	def materialize(self, package, attrs):
		"""
		Materializes the package data into the database.
		"""
		if package != self.current_package:
			if isinstance(package, SequenceBasedPackage):
				self._materialize_seqbased(package, attrs)

			elif isinstance(package, InMemoryPackage):
				self._materialize_list(package, attrs)

			elif isinstance(package, list):
				self._materialize_list(package, attrs)

			else:
				raise Exception("Unknown package data model: {}".format(package.__class__.__name__))

			self.current_package = package


	@abstractmethod
	def delete(self):
		raise NotImplementedError


	@abstractmethod
	def drop(self):
		raise NotImplementedError


	@abstractmethod
	def _materialize_seqbased(self, package, attrs):
		raise NotImplemented


	@abstractmethod
	def _materialize_list(self, package, attrs):
		raise NotImplemented


	def get_coretable_size(self):
		return len(self.coretable)


	@abstractmethod
	def destroy(self):
		raise NotImplemented

