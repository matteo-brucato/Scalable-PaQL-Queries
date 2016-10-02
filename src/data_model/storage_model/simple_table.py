import sys

from src.data_model.storage_model.__package import PackageStorageModel
from src.data_model.memory_representation.data_in_memory import InMemoryPackage
from src.data_model.memory_representation.sequence_based import SequenceBasedPackage
from src.data_model.tuple import Tuple
from src.utils.log import log



class SimpleTableDataModel(PackageStorageModel):
	"""
	In this implementation, a package is a simple table with the same schema as the input table.

	When the materialize() function is called, this table is filled with actual tuples copied from the input table,
	according to a given in-memory package representation.

	Each object of this class creates a unique *temporary* table in the database to materialize a package into.
	"""

	candidate_table_counter = 0


	def __init__(self, *args):
		super(SimpleTableDataModel, self).__init__(*args)

		print "Initing Simple Candidate Table..."

		self.table_name = "simple_candidate_package_{}".format(SimpleTableDataModel.candidate_table_counter)
		SimpleTableDataModel.candidate_table_counter += 1

		self._db.sql_update(
			"CREATE TEMP TABLE {candidate_table} AS \n"
			"SELECT {attrs} \n"
			"FROM {S}.{D} C \n"
			"WITH NO DATA".format(
				candidate_table=self.table_name,
		        attrs="*",
				S=self.coretable.schema_name,
				D=self.coretable.table_name))

		self.data_attributes = sorted(set(
			c.column_name for c in self._db.sql_query(
				"SELECT column_name "
				"FROM information_schema.columns "
				"WHERE table_name=%s ",
				self.table_name)))

		log("Simple Candidate Table inited.")


	def get_str_list(self):
		s = super(SimpleTableDataModel, self).get_str_list()
		s.append("Candidate table counter: {}".format(self.candidate_table_counter))
		return s


	def __str__(self):
		s = [ "Simple Table Package Data Model:" ]
		s.extend(self.get_str_list())
		return "\n".join(s)


	def get_core_table(self):
		raise DeprecationWarning
		sql = "SELECT C.* FROM {core_table} C ".format(
			core_table=self.coretable.table_name)
		return self._db.sql_query(sql)


	def delete(self):
		log("deleting package table...")
		self._db.sql_update(
			"DELETE FROM {candidate_table}".format(
				candidate_table=self.table_name))


	def drop(self):
		log("dropping package table...")
		self._db.sql_update(
			"DROP TABLE {candidate_table}".format(
				candidate_table=self.table_name))


	def _materialize_seqbased(self, package, attrs):
		"""
		:param package: A SequenceBasedPackage package
		:param attrs: A collection of attributes
		"""
		assert isinstance(package, SequenceBasedPackage)

		if attrs is None:
			use_attrs = { "id" } | self._package_query.get_attributes()
		elif attrs == "*":
			use_attrs = self.data_attributes
		else:
			use_attrs = { "id" }

		self.delete()

		log("materializing package table (seqbased)...")
		self._db.sql_update(
			"INSERT INTO {candidate_table} ({attrs}) \n"
			"SELECT {attrs} FROM (\n\t"
			"	SELECT {attrs}, ROW_NUMBER() OVER (ORDER BY id) AS seq \n\t"
			"	FROM {schema_name}.{core_table}) AS R \n"
			"WHERE seq IN %s".format(
				candidate_table=self.table_name,
				schema_name=self.coretable.schema_name,
				core_table=self.coretable.table_name,
				attrs=", ".join(use_attrs)
			),
			tuple(package.combination))


	def _materialize_list(self, package, attrs):
		"""
		:param package: A list of Tuples
		:param attrs: A collection of attributes
		"""
		assert isinstance(package, list) or isinstance(package, InMemoryPackage)

		if attrs is None:
			if isinstance(package, InMemoryPackage):
				use_attrs = package.attrs
			else:
				use_attrs = { "id" } | self._package_query.get_attributes()
		elif attrs == "*":
			use_attrs = self.data_attributes
		else:
			use_attrs = { "id" }

		self.delete()

		log("list: creating package table...")
		for t in package:
			assert isinstance(t, Tuple)
			self._db.sql_update(
				"INSERT INTO {candidate_table} ({attrs}) VALUES ({vals})".format(
					candidate_table=self.table_name,
					attrs=", ".join("{}".format(a) for a in use_attrs),
					vals=", ".join("{}".format(getattr(t, a)) for a in use_attrs)
				))


	def get_seqs_by_ids(self, ids):
		"""
		Returns the internal sequence used to identify tuples that have the specified ids.
		"""
		# TODO: Move this method in sequence based package class.
		return [
			r.seq for r in
			self._db.sql_query(
				"SELECT seq FROM (\n"
				"	SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS seq \n"
				"	FROM {schema_name}.{core_table}) AS R \n"
				"WHERE id = ANY(%s)".format(
					schema_name=self.coretable.schema_name,
					core_table=self.coretable.table_name),
			ids)
		]


	def destroy(self):
		self.drop()
		self.coretable.destroy()
		self._db.commit()
