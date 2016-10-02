import psutil
from src.dbms.dbsettings import dbsettings

from src.dbms.db import *
from src.paql_eval.sketch_refine.sketch_refine import Tuple



def main():
	proc = psutil.Process(os.getpid())

	print "INIT MEM:", proc.memory_info()[0] / float(2 ** 20)

	for cursor_type in ["namedtuple",]:# [ None,  ]:# "dictlike", "dict" ]:
		conn = DBConnection(cursor_type=cursor_type, **dbsettings)
		db = DBCursor(conn)
		print "Cursor Type:", db._psql_cur.__class__.__name__


		rtime = -time.time()
		r = db.sql_query(
			"SELECT D.id, D.p_size, D.ps_min_supplycost "
			"FROM q2_20p D "
			"WHERE D.cid = 1 "
			"ORDER BY D.id")
			# "SELECT id,cid,p_size,ps_min_supplycost "
			# "FROM q2_20p "
			# "WHERE cid = 1 "
			# "ORDER BY id")
		n = 0
		for t in r:
			# print t.p_size
			n += 1
			Tuple(kind="Tuple", attrs=["id", "p_size", "ps_min_supplycost"], record=t)
		rtime += time.time()
		print n, rtime
		exit()









		db.sql_update("CREATE TEMP TABLE x_temp_table AS SELECT * FROM galaxies WITH NO DATA")
		ids = list(db.sql_query("INSERT INTO x_temp_table SELECT * FROM galaxies LIMIT 10 RETURNING id"))
		print ids

		for i in xrange(2):
			# r = dbms.sql_query("SELECT * FROM galaxies")
			r = db.sql_query("SELECT * FROM x_temp_table")
			print "AFTER QUERY EXEC MEM:", proc.memory_info()[0] / float(2 ** 20)

			rtime = -time.time()
			for l in r: print l.id,
			rtime += time.time()
			print

			ids = list(db.sql_query("INSERT INTO x_temp_table SELECT * FROM galaxies LIMIT 1 RETURNING id"))
			print ids

			print "AFTER ALL MEM:", proc.memory_info()[0] / float(2 ** 20)
			print "RTIME:", rtime
			print "=" * 100

		ids = list(db.sql_query("DELETE FROM x_temp_table RETURNING id"))
		print ids

		db.sql_update("DROP TABLE x_temp_table")

		conn.close()


if __name__ == "__main__":
	main()
