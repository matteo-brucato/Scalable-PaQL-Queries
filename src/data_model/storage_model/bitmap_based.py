import sys
from bitarray import bitarray

from src.data_model.storage_model.__package_data_model import PackageDataModel
from src.utils.log import log
from src.utils.utils import pretty_table_str_named




class BitmapBasedDataModel(PackageDataModel):


	candidate_table_counter = 0


	@property
	def table_name(self):
		return self._tablename


	@table_name.setter
	def table_name(self, tablename):
		self._tablename = tablename


	def __init__(self, *args):
		super(BitmapBasedDataModel, self).__init__(*args)

		self.tablename = "bitmap_candidate_package_{}".format(BitmapBasedDataModel.candidate_table_counter)
		BitmapBasedDataModel.candidate_table_counter += 1

		print "done. Initing bitmap table...", ; sys.stdout.flush()
		self.bitmaptable = BitmapTable(self._db, self.coretable)
		print "done.", ; sys.stdout.flush()

		# The candidate table is a view which joins bitmap and core tables where inset=true
		query_attrs = self._package_query.get_attributes()
		sql = ("CREATE TEMP VIEW {package_table} AS \n"
		       "SELECT C.id, {attrs} B.seq \n"
		       "FROM {core_table} C, {bitmap_table} B \n"
		       "WHERE C.id = B.id AND B.inset = true".format(
			package_table=self.table_name,
		    attrs=(",".join("C.{}".format(a) for a in query_attrs) + ",") if len(query_attrs) > 0 else "",
			core_table=self.coretable.table_name,
			bitmap_table=self.bitmaptable.tablename))
		self._db.sql_update(sql)


	def get_core_table(self):
		sql = (
			"SELECT C.*, B.seq FROM {core_table} C, {bitmap_table} B "
			"WHERE C.id = B.id "
			"ORDER BY B.seq".format(
				core_table=self.coretable.table_name,
				bitmap_table=self.bitmaptable.tablename))
		return self._db.sql_query(sql)


	def _materialize_seqbased(self, package, attrs):
		print ">>>"
		self.bitmaptable.update(package)
		print "<<<"


	def _materialize_list(self, package, attrs):
		raise NotImplementedError


	def get_seqs_by_ids(self, ids):
		"""
		Returns the internal sequence ids used by the bitmap table which correspond to the tuple ids passed in.
		"""
		return self.bitmaptable.get_seqs_by_ids(ids)


	def destroy(self):
		self._db.sql_update("DROP VIEW IF EXISTS {data_models}".format(candidate_table=self.table_name))
		self.bitmaptable.destroy()
		self.coretable.destroy()
		self._db.commit()




class BitmapTable(object):
	bitmap_table_counter = 0

	@property
	def tablename(self):
		return self._bitmap_table_name


	@tablename.setter
	def tablename(self, tablename):
		self._bitmap_table_name = tablename


	def __init__(self, db, coretable):
		self._bitmap_table_name = None

		self.coretable = coretable
		self.tablename = "bitmap_table_{}".format(BitmapTable.bitmap_table_counter)
		BitmapTable.bitmap_table_counter += 1

		# TODO: Try to use this query: SELECT *, ROW_NUMBER() OVER (ORDER BY id) AS seq

		self._db = db
		self._db.sql_update(
			"CREATE GLOBAL TEMP TABLE {bitmap_table_name}("
			"   seq serial NOT NULL,"
			"   id varchar(100) NOT NULL,"
			"   inset boolean DEFAULT false NOT NULL "
			") "
			"ON COMMIT PRESERVE ROWS".format(
				bitmap_table_name=self.tablename))

		print "Inserting id's into bitmap table...", ; sys.stdout.flush()
		self._db.sql_update("INSERT INTO {bitmap_table_name}(id) SELECT id FROM {core_table_name} ORDER BY id".format(
			bitmap_table_name=self.tablename,
			core_table_name=self.coretable.table_name))

		print "Setting id as primary key of bitmap table...", ; sys.stdout.flush()
		self._db.sql_update("ALTER TABLE {bitmap_table_name} ADD PRIMARY KEY (id)".format(
			bitmap_table_name=self.tablename))

		print "Creating index on seq of bitmap table...", ; sys.stdout.flush()
		self._db.sql_update("CREATE INDEX ON {bitmap_table_name}(seq)".format(
			bitmap_table_name=self.tablename))

		# Store bitmap in main memory to allow fast update of the bitmap table (to update only modified tuples)
		print "Creating in-memory bitmap...", ; sys.stdout.flush()
		self.bitmap = bitarray(len(self.coretable))
		self.bitmap.setall(0)
		# self.last_updated_package = None

		# Store an inverted index from tuple ids to sequence ids
		print "Creating in-memory id->seq index...", ; sys.stdout.flush()
		self._id_to_seq = {}
		r = self._db.sql_query("SELECT B.seq, B.id FROM {bitmap_table_name} B".format(bitmap_table_name=self.tablename))
		for seq, _id in r:
			self._id_to_seq[_id] = seq


	def update(self, package):
		log("updating bitmap...")

		new_bitmap = bitarray(len(self.coretable))
		new_bitmap.setall(0)
		for seq in package.combination:
			new_bitmap[seq - 1] = 1
		assert len(new_bitmap) == len(self.bitmap)

		change_bits = new_bitmap ^ self.bitmap
		start, true_seqs, false_seqs = 0, [] ,[]
		while True:
			try:
				change_bit = change_bits.index(1, start)
			except ValueError:
				break
			else:
				# print change_bit
				if new_bitmap[change_bit]:
					true_seqs += [change_bit + 1]
				else:
					false_seqs += [change_bit + 1]
				start = change_bit + 1

		if len(true_seqs) > 0:
			change_bit_query = "UPDATE {} SET inset = true WHERE seq = ANY(ARRAY[{}])".format(
				self.tablename,
				",".join([ "{:d}".format(i) for i in true_seqs ]))
			self._db.sql_update(change_bit_query)

		if len(false_seqs) > 0:
			change_bit_query = "UPDATE {} SET inset = false WHERE seq = ANY(ARRAY[{}])".format(
				self.tablename,
				",".join([ "{:d}".format(i) for i in false_seqs ]))
			self._db.sql_update(change_bit_query)

		self.bitmap = new_bitmap

		log("updating bitmap: done.")


	def get_seqs_by_ids(self, ids):
		"""
		Returns the internal sequence ids used by the bitmap table which correspond to the tuple ids passed in.
		"""
		return (self._id_to_seq[_id] for _id in ids)


	def printout(self):
		res = self._db.sql_query("SELECT * FROM {} ORDER BY seq".format(self.tablename))
		print pretty_table_str_named(res)


	def destroy(self):
		self._db.sql_update("DROP TABLE {}".format(self.tablename))



