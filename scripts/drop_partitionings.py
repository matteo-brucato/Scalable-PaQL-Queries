import sys

from mpmath import isint

from src.dbms.db import DBConnection, DBCursor
from src.dbms.dbms_settings import data_dbms_settings


def drop_tables(db, dropping_tables):
	for relname in dropping_tables:
		db.sql_update("DROP TABLE sketchrefine.{}".format(relname))
	print "All tables dropped."
	print "Committing..."
	db.commit()
	print "Done."


def ask_drop(db, tables, name):
	if tables:
		print "Existing stored {} tables:\n{}\n".format(
			name, "\n".join("{}. {}".format(i + 1, t) for i, t in enumerate(tables)))

		if len(tables) > 0:
			cont = raw_input("{} stored {} tables to drop. Drop all? Y/[n] or list of comma-separated numbers ".format(
				len(tables), name))

			if cont=="Y":
				drop_tables(db, tables)
				return

			else:
				selections = cont.split(",")
				if all(isint(s) and int(s) >= 1 for s in selections):
					drop_tables(db, [ tables[int(s) - 1] for s in selections ])
					return

			print "No {} tables were dropped.".format(name)

	else:
		print "There are no {} tables.".format(name)


def main():
	dbconn = DBConnection(**data_dbms_settings)
	db = DBCursor(dbconn)

	all_partitioning_tables = [
		rn.table_name for rn in db.sql_query(
			"SELECT table_name "
			"FROM information_schema.tables "
			"WHERE table_schema = 'sketchrefine' "
			"AND table_type = 'BASE TABLE' AND table_name LIKE 'clus_%%'")
	]

	all_representative_tables = [
		rn.table_name for rn in db.sql_query(
			"SELECT table_name "
			"FROM information_schema.tables "
			"WHERE table_schema = 'sketchrefine' "
			"AND table_type = 'BASE TABLE' AND table_name LIKE 'repr_%%'")
	]

	all_partitioning_indexes = [
		rn.table_name for rn in db.sql_query(
			"SELECT table_name "
			"FROM information_schema.tables "
			"WHERE table_schema = 'sketchrefine' "
			"AND table_type = 'BASE TABLE' AND table_name LIKE 'idx_%%'")
	]

	# PARTITIONING TABLES
	ask_drop(db, all_partitioning_tables, "partitioning")

	# REPRESENTATIVE TABLES
	ask_drop(db, all_representative_tables, "representative")

	# INDEX TABLES
	ask_drop(db, all_partitioning_indexes, "partitioning index")

	# Partitioning columns from data tables
	partitioned_data_tables = [
		(rn.table_name, rn.column_name) for rn in db.sql_query(
			"SELECT table_name, column_name "
			"FROM information_schema.columns C "
			"WHERE table_schema = 'public' AND column_name = 'cid'"
			"  AND table_name NOT LIKE 'clus_%%'"
			"  AND table_name NOT LIKE 'repr_%%'"
			"  AND table_name NOT LIKE 'idx_%%'")
	]

	print "Dropping partitioning columns from clustered data tables:\n{}".format(
		"\n".join("{} from table {}".format(t[1], t[0]) for t in partitioned_data_tables))

	if len(partitioned_data_tables) > 0:
		cont = raw_input("{} partitioning columns {} to drop. Continue? Y/[n] ".format(
			len(partitioned_data_tables),
			",".join(r[1] for r in all_partitioning_tables),
		))
		if cont == "Y":
			for relname, colname in partitioned_data_tables:
				db.sql_update("ALTER TABLE {} DROP COLUMN cid".format(relname))
				db.commit()

				try:
					db.sql_update("ALTER TABLE {} DROP COLUMN node_level".format(relname))
				except:
					db.commit()

				try:
					db.sql_update("ALTER TABLE {} DROP COLUMN cid_size".format(relname))
				except:
					db.commit()

			print "All partitioning columns dropped from data tables."

			print "Deleting all from current partitioning table:"
			db.sql_update("DELETE FROM sketchrefine.currently_loaded_partitioning")
			print "Table 'currently_loaded_partitioning' emptied."

			print "Committing...", ; sys.stdout.flush()
			db.commit()
			print "done."




if __name__ == "__main__":
	main()
