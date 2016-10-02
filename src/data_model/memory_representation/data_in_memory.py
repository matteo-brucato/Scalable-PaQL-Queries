import random

from src.dbms.db import DBCursor
from src.dbms.utils import get_db_type, to_db_val
from src.data_model.memory_representation.__package import Package
from src.data_model.tuple import Tuple
from src.paql.package_query import PackageQuery
from src.utils.utils import pretty_table_str, int_round



class InMemoryPackage(Package):

	def __init__(self, search=None, tuples=None, attrs=None, package_query=None, *args, **kwargs):
		super(InMemoryPackage, self).__init__(search, *args, **kwargs)

		# If this package satisfies a known PaQL
		assert package_query is None or isinstance(package_query, PackageQuery)

		self.package_query = package_query
		self._attrs = None
		self._tuples = []

		assert (tuples is not None and attrs is None) or (tuples is None and attrs is not None)

		if tuples is None:
			self.attrs = attrs

		else:
			for t in tuples:
				self.append(t)


	@staticmethod
	def from_table_name(table_name, db, *args, **kwargs):
		assert isinstance(db, DBCursor)
		db.commit()
		tuples = []
		for r in db.sql_query("SELECT * FROM {}".format(table_name)):
			tuples.append(Tuple(r))
		return InMemoryPackage(tuples=tuples, table_name=table_name, *args, **kwargs)


	@property
	def attrs(self):
		return self._attrs


	@attrs.setter
	def attrs(self, value):
		self._attrs = value


	def __iter__(self):
		return self.iter_tuples()


	def __len__(self):
		return len(self._tuples)


	def pretty_str(self, footers=None):
		return pretty_table_str(self._tuples, header=self.attrs, footers=footers)


	def __str__(self):
		return self.pretty_str()


	def materialize(self, attrs=None, db=None, schema_name=None, table_name=None, temp=True):
		if self.search is not None:
			self.search.package_table.materialize(self, attrs)

		else:
			attrs_dbtypes = [
				get_db_type(self._tuples[0][attr])
				for attr in self.attrs
			]

			schema_name = schema_name + "." if schema_name is not None else ""

			db.sql_update(
				"CREATE {temp} TABLE {schema_name}{table_name} ({attrs})".format(
					temp="TEMP" if temp else "",
					schema_name=schema_name,
					table_name=table_name,
					attrs=",".join("{} {}".format(self.attrs[i], attrs_dbtypes[i])
					               for i in xrange(len(self.attrs)))))

			# FIXME: You can make it faster by using executemany from psycopg2 (there's a flag for it in sql_update())
			for t in self:
				db.sql_update(
					"INSERT INTO {schema_name}{table_name} ({attrs}) VALUES ({vals})".format(
						schema_name=schema_name,
						table_name=table_name,
						attrs=",".join(self.attrs),
						vals=",".join("{}".format(to_db_val(t[attr])) for attr in self.attrs)))

			self.table_name = "{}{}".format(schema_name, table_name)


	def delete(self):
		self.search.package_table.delete()


	def drop(self, db, table_name, schema_name=None):
		self.search.package_table.drop()


	def perturb_mult(self):
		for t in self:
			for attr in self.attrs:
				rand_sign = random.choice([ 1, -1 ])
				rand_mult = random.random()
				new_val = getattr(t, attr) + (rand_sign * rand_mult * getattr(t, attr))
				if new_val > 5:
					new_val = int_round(new_val, base=5)
				setattr(t, attr, new_val)


	@staticmethod
	def generate_random_candidate(search):
		raise NotImplementedError


	def iter_tuples(self, attrs=None):
		for t in self._tuples:
			if attrs is not None:
				t = Tuple(attrs=attrs, record=t)
			yield t


	def append(self, new_tuple):
		assert isinstance(new_tuple, Tuple)
		if self.attrs is None:
			self.attrs = sorted(set(new_tuple.attrs))
		else:
			pass
		self._tuples.append(new_tuple)


	def get_all_global_scores(self):
		"""
		Return all possible global scores that we care about (cardinality, and sums on all attributes).
		"""
		global_scores = {}
		for t in self:
			global_scores["count", "*"] = global_scores.get(("count", "*"), 0) + 1
			for attr in self.attrs:
				val = getattr(t, attr)
				if val is not None and type(val) is int or type(val) is float:
					global_scores["sum", attr] = global_scores.get(("sum", attr), 0.0) + val
		return global_scores


	def get_objective_value(self, package_query):
		assert isinstance(package_query, PackageQuery)
		obj_val = sum(getattr(t, obj_attr)
		              for obj_attr in package_query.get_objective_attributes()
		              for t in self)
		return obj_val
