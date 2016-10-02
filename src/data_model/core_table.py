from src.dbms.db import DBCursor
from src.dbms.utils import sql_table_exists, sql_get_all_indexes
from src.paql.package_query import PackageQuery
from src.utils.log import *


class CoreTable(object):
	"""
	The core table is the base table after applying the base constraints.
	"""

	core_table_counter = 0

	@property
	def table_name(self):
		return self._core_table_name


	@table_name.setter
	def table_name(self, tablename):
		self._core_table_name = tablename


	def __init__(self, db, query):
		assert isinstance(db, DBCursor)
		assert isinstance(query, PackageQuery)

		log("Initing core table...")

		assert isinstance(db, DBCursor)
		assert isinstance(query, PackageQuery)
		self._core_table_name = None # The name of the core table
		self._size = None
		self._db = db

		# Apply base constraints
		if len(query.uncoalesced_bcs) > 0:
			# Create an actual core_table only if some base constraints are present
			self.schema_name = "core_tables"
			self.table_name = "{}_{}_{}".format(
				query.table_name,
				query.md5(),
				CoreTable.core_table_counter)
			CoreTable.core_table_counter += 1

			if not sql_table_exists(db, self.schema_name, self.table_name):

				log("Evaluating base constraints...", query.uncoalesced_bcs)
				db.sql_update("CREATE TABLE {S}.{D} "
				              "AS {base_constraints_sql}".format(
					S=self.schema_name,
					D=self.table_name,
					base_constraints_sql=query.bc_query))

				log("Adding primary key on (id) to core table...")
				db.sql_update("ALTER TABLE {S}.{D} ADD PRIMARY KEY (id)".format(
					S=self.schema_name,
					D=self.table_name))

			else:
				log("Core table already exists: Doing nothing.")

		else:
			# Otherwise the core table is the base table itself (no initial selection)

			if "." in query.table_name:
				split = query.table_name.split(".")
				assert len(split) == 2
				self.schema_name = split[0].lower()
				self.table_name = split[1].lower()
			else:
				self.schema_name = "public"
				self.table_name = query.table_name.lower()

		print "Core table info: schema={}, table={}".format(self.schema_name, self.table_name)

		db.commit()

		# Make sure the core table has a primary key
		indexes = sql_get_all_indexes(db, self.schema_name, self.table_name)
		for index, colname, isprimary in indexes:
			print index, colname, isprimary
			if isprimary: break
		else:
			raise Exception("The core table '{}.{}' does not have an index on the primary key "
			                "or it does not have a primary key.".format(
				self.schema_name,
				self.table_name))

		log("Done: core table init.")


	def __len__(self):
		if self._size is None:
			self._size = self._db.sql_query(
				"SELECT COUNT(*) FROM {}.{}".format(
					self.schema_name,
					self.table_name)).next()[0]
			assert type(self._size) is long and self._size >= 0
		return self._size


	def destroy(self):
		if self.schema_name == "core_tables":
			self._db.sql_update("DROP TABLE {}".format(self.table_name))
		else:
			warn("Ignoring destroy() of core table not in schema 'core_tables'.")
