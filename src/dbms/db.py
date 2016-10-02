import os
import traceback
import sys
import time
import uuid
from operator import xor

import psycopg2 as pgres
from psycopg2._psycopg import OperationalError
from psycopg2.extensions import QueryCanceledError, ISOLATION_LEVEL_READ_UNCOMMITTED
from psycopg2.extras import NamedTupleCursor, DictCursor, RealDictCursor
from sshtunnel import SSHTunnelForwarder

from src.paql_eval.exceptions import TimeLimitElapsed
from src.utils.log import warn, debug, log



class DBConnection:
	"""
	This can be global and shared among different threads.
	"""

	named_cursor_n = -1


	def __init__(self, dbname, username, password, host, port, cursor_type=None, verbose=False):
		self.sshtunnel = None
		self._psql_conn = None
		self.verbose = verbose
		# self.connection_string = (
		# 	"dbname={} user={} "
		# 	"password={} " if password is not None else "{}"
		# 	"host={} " if host is not None else "{}"
		# 	"port={} " if host is not None else "{}").format(
		# 	dbname,
		# 	username,
		# 	password if password is not None else "",
		# 	host if host is not None else "",
		# 	port if port is not None else "")

		self.dbname = dbname
		self.username = username
		self.password = password
		self.host = host
		self.port = int(port) if port != "" else 5432

		# self.connection_string = "dbname={} ".format(dbname)
		# self.connection_string += "user={} ".format(username)
		# self.connection_string += "password={} ".format(password) if password is not None else ""
		# self.connection_string += "host={} ".format(host) if host is not None else ""
		# self.connection_string += "port={}".format(port) if host is not None else ""
		#
		# print self.connection_string

		if cursor_type is None:
			self.cursor_factory = NamedTupleCursor
		elif cursor_type == "standard":
			self.cursor_factory = None
		elif cursor_type == "namedtuple":
			self.cursor_factory = NamedTupleCursor
		elif cursor_type == "dictlike":
			self.cursor_factory = DictCursor
		elif cursor_type == "dict":
			self.cursor_factory = RealDictCursor
		else:
			raise Exception("'cursor_type' unknonw: '{}'".format(cursor_type))

		# self.conn = pgres.connect(connection_string, cursor_factory=NamedTupleCursor)
		# self._conn = pgres.connect(connection_string)

		self.connect(self.cursor_factory)


	def commit(self):
		# print self.get_transaction_status_string()
		self._psql_conn.commit()
		# self._psql_conn.set_isolation_level(ISOLATION_LEVEL_READ_UNCOMMITTED)


	def abort(self):
		self._psql_conn.rollback()


	def connect(self, cursor_factory=None):
		if self.sshtunnel is not None:
			self.sshtunnel.close()

		if self.host != "localhost" and self.host != "127.0.0.1" and self.host != "":
			self.sshtunnel = SSHTunnelForwarder((self.host, 22), remote_bind_address=("127.0.0.1", self.port))
			self.sshtunnel.start()
			# conn = pgres.connect(database="nnexp", port=server.local_bind_port, host="localhost")

			connection_info = {
				"database": self.dbname,
				"user": self.username,
				"password": self.password,
				"host": "localhost",
				"port": self.sshtunnel.local_bind_port,
			}

		else:
			connection_info = {
				"database": self.dbname,
				"user": self.username,
				"password": self.password,
				"host": self.host,
				"port": self.port,
			}

		# print connection_info

		n_retrials = 10
		for i in xrange(n_retrials):
			try:
				self._psql_conn = pgres.connect(cursor_factory=cursor_factory, **connection_info)

			except OperationalError:
				sleep_time = 0.02 * (2 ** i)
				warn("Cound not connect to DB ({}), retrying in {} seconds...".format(connection_info, sleep_time))
				time.sleep(sleep_time)

			else:
				break

		else:
			raise Exception("Could not connect to the DB after {} trials".format(n_retrials))


	def reconnect(self):
		self.connect()


	def get_cursor(self, named=True):
		# An "unnamed" cursor, i.e., without specifying a name, alwyas fetches all the data from the DB!
		# return self._psql_conn.cursor()

		# named = False

		if named:
			# This one instead, fetches only the data that you want
			self.named_cursor_n += 1
			return self._psql_conn.cursor(name="cursor_{}{}{}".format(os.getpid(), time.time(), self.named_cursor_n))
		else:
			return self._psql_conn.cursor()


	def close(self):
		# NOTE: This close always commits before closing (differently from standard behaviour that aborts if not committed)
		if not self._psql_conn.closed:
			self._psql_conn.commit()
			self._psql_conn.close()

		if self.sshtunnel is not None:
			self.sshtunnel.close()
			self.sshtunnel = None


	def __del__(self):
		self.close()


	def get_transaction_status_string(self):
		status = self._psql_conn.get_transaction_status()

		if status == pgres.extensions.TRANSACTION_STATUS_IDLE:
			return "TRANSACTION_STATUS_IDLE"

		elif status == pgres.extensions.TRANSACTION_STATUS_ACTIVE:
			return "TRANSACTION_STATUS_ACTIVE"

		elif status == pgres.extensions.TRANSACTION_STATUS_INERROR:
			return "TRANSACTION_STATUS_INERROR"

		elif status == pgres.extensions.TRANSACTION_STATUS_INTRANS:
			return "TRANSACTION_STATUS_INTRANS"

		elif status == pgres.extensions.TRANSACTION_STATUS_UNKNOWN:
			return "TRANSACTION_STATUS_UNKNOWN"



class DBCursor:
	"""
	This *cannot* be shared among threads! So, it may not be global.
	Use its objects to acually issue queries to the DB.
	"""

	@property
	def dbname(self):
		return self.connection.dbname


	def __init__(self, connection, logfile=None, sqlfile=None):
		self._psql_cur = None
		self._psql_unnamed_cur = None

		assert isinstance(connection, DBConnection)
		self.connection = connection
		self.verbose = self.connection.verbose

		assert logfile is None or type(logfile) is file

		# All read queries run through this cursor, and the number of records retrieved
		self._queries = []  # list of tuples <query, status, n_records_retrieved>

		# All write queries run through this cursor
		self._update_queries = []

		self.logfile = logfile
		self.sqlfile = sqlfile

		if self.sqlfile is not None:
			self.sqlfile.write("START TRANSACTION;\n\n")

		self.cursor_id = uuid.uuid1()

		self.last_select_id = 0
		self.last_update_id = 0

		self.last_select_query = None
		self.last_update_query = None

		self.create_cursors()


	def create_cursors(self):
		# This is for inserts, updates and deletes
		self._psql_unnamed_cur = self.connection.get_cursor(named=False)

		# Get cursor from connection object
		# self._psql_cur = self._connection.get_cursor()


	def commit(self):
		# print "TABLES BEFORE COMMIT: ", \
		# self.sql_query("SELECT table_name, table_schema, table_catalog FROM information_schema.tables "
		#                      "WHERE table_name LIKE '%core%' OR table_name LIKE '%bitmap%';")
		self.connection.commit()


	def close(self):
		if self._psql_cur is not None and not self._psql_cur.closed:
			# print self.dbname, self._psql_cur
			try:
				self._psql_cur.close()
			except Exception as e:
				print "Exception: {}".format(e)
				print "Probably you didn't extract all rows from a previous SQL query."
				raise e
		self._psql_unnamed_cur.close()
		self.connection.close()


	def abort(self):
		self.connection.abort()


	def set_timelimit_sec(self, timelimit_sec):
		assert timelimit_sec is None or timelimit_sec >= 0
		if timelimit_sec is not None:
			# Note: with SET LOCAL we set the configuration only on the current *transaction* (not session/connection!)
			# print self.sql_query("SHOW statement_timeout;")[0][0]
			self.sql_update("SET LOCAL statement_timeout TO {};".format(int(timelimit_sec * 1000)))
		# print self.sql_query("SHOW statement_timeout;")[0][0]
		else:
			# TODO: This line may be redundant in case you don't really change the time limit during the paql_eval.
			self.sql_update("SET LOCAL statement_timeout TO 0;")


	def rowcount(self):
		return self._psql_cur.rowcount


	def mogrify(self, string, params):
		cur = self.connection.get_cursor(named=False)
		result = cur.mogrify(string, params)
		cur.close()
		return result


	def _execute_query(self, sql, args):
		# sql = sql.lower().strip()
		# print sql
		sql_strip = sql.lower().strip()
		# print self.dbname, sql_strip
		if sql_strip.startswith("select ") or \
				(sql_strip.startswith("with ")
				 # and "update " not in sql_strip and "insert " not in sql_strip
				 ):
			# Try to close previous named cursor
			# if self._psql_cur is not None and not self._psql_cur.closed:
			# try:
			# 	self._psql_cur.close()
			# except ProgrammingError:
			# 	pass

			# self._psql_cur.scroll(self._psql_cur.rowcount, mode="absolute")
			# self._psql_cur.fetchone()
			# self._psql_cur.fetchone()

			# Create a new named cursor
			self._psql_cur = self.connection.get_cursor()

			# print self.dbname, "NAMED", self._psql_cur

			# Execute query
			self._psql_cur.execute(sql, args)
			return self._psql_cur, True
		else:
		# if "insert " in sql or "update " in sql or "delete " in sql or "create" in sql:
		# 	print self.dbname, "UNNAMED"
			# In this case, do not use the named (server side) cursor
			# self._psql_unnamed_cur = self._connection.get_cursor(named=False)
			self._psql_unnamed_cur.execute(sql, args)
			return self._psql_unnamed_cur, False


	def __call__(self, *args, **kwargs):
		return self.sql_query(*args, **kwargs)


	def sql_query(self, sql, *data, **kwdata):
		"""
		NOTE: This function returns a generator. So if you use it to do any kind of update to the dbms that doesn't
		return anything, it won't be executed!
		"""
		# print ("%"*50) + " QUERY " + ("%"*50)
		self.last_select_id += 1

		n_retrials = kwdata.get("___n_retrials", 0)
		if n_retrials > 10:
			raise OperationalError

		assert not (len(data) > 0 and len(set(kwdata) - {"___n_retrials"}) > 0), \
			"Pass either keyword-based data or comma-separated data."

		time_start = time.time()
		n_records_retrieved = 0
		status = None
		toclose = False

		if self.logfile is not None:
			self.logfile.write(">>> {} {} {} START SELECT\n{}\ndata={}\nkwdata={}\n\n".format(
				self.cursor_id, self.last_select_id, time_start, sql, data, kwdata))

		# print "\n*** QUERY:", sql, "\n"

		try:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> QUERY..."
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> QUERY..."
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> QUERY..."
			if len(data) > 0:
				# self._psql_cur.execute(sql, data)
				cur, toclose = self._execute_query(sql, data)
			elif len(kwdata) > 0:
				# self._psql_cur.execute(sql, kwdata)
				cur, toclose = self._execute_query(sql, kwdata)
			else:
				cur, toclose = self._execute_query(sql, None)
			n_records_reported = cur.rowcount
			# print n_records_reported
			# Yield records
			for record in cur:
				n_records_retrieved += 1
				if n_records_retrieved == n_records_reported:
					status = "Finished"
				yield record

		# except KeyboardInterrupt:
		# 	self._connection._psql_conn.cancel()  # FIXME: Make it general
		# 	raise KeyboardInterrupt

		except QueryCanceledError:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CANCEL Q"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CANCEL Q"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CANCEL Q"
			debug("QueryCanceledError")
			status = "QueryCanceledError"
			raise TimeLimitElapsed

		except OperationalError as e:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OP Q"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OP Q"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OP Q"
			self.connection.reconnect()
			self.create_cursors()
			kwdata["___n_retrials"] = n_retrials + 1
			self.sql_query(sql, *data, **kwdata)

		except pgres.Error as e:
			print '=' * 60
			print "Exception occured while executing query:\n{}\n".format(sql)
			traceback.print_exc(file=sys.stdout)
			print e.diag.message_primary
			print e.pgerror
			print "pgcode: ", e.pgcode
			status = "pgcode: {}".format(e.pgcode)
			raise e

		except Exception as e:
			print '=' * 60
			print "Exception occured while executing query:\n{}\n".format(sql)
			traceback.print_exc(file=sys.stdout)
			print e
			raise e

		else:
			status = "Finished"
			if self.verbose:
				print cur.query
				print cur.statusmessage
			self.last_select_query = cur.query
			if toclose and not cur.closed:
				cur.close()

		finally:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FINALLY Q"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FINALLY Q"
			# print sql
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FINALLY Q"

			# print ("%"*50) + "%%%%%%%" + ("%"*50)
			time_end = time.time()

			# if self.logfile:
			# 	self._queries.append((time_start, time_end, status, sql, data, kwdata, n_records_retrieved))

			if self.logfile is not None:
				log("Writing to log file '{}'...".format(self.logfile.name))
				self.logfile.write(">>> {} {} {} END SELECT\n{} - {} record(s) retrieved\n\n".format(
					self.cursor_id, self.last_select_id, time_end, status, n_records_retrieved))
				self.logfile.flush()

			if self.sqlfile is not None:
				log("Writing query to SQL file '{}'...".format(self.sqlfile.name))
				self.sqlfile.write("-- {} select\n".format(self.last_select_id))
				self.sqlfile.write(cur.query + ";")
				self.sqlfile.write("\n\n")
				self.sqlfile.flush()


	def sql_update(self, sql, *data, **kwdata):
		# print ("%"*50) + " UPDATE " + ("%"*49)
		self.last_update_id += 1

		self._psql_unnamed_cur = self.connection.get_cursor(named=False)

		n_retrials = kwdata.get("___n_retrials", 0)
		if n_retrials > 10:
			raise OperationalError

		# assert xor(xor(len(data) > 0, len(set(kwdata) - {"___n_retrials"}) > 0), _many_iter_data is not None), \
		assert not (len(data) > 0 and len(set(kwdata) - {"___n_retrials"}) > 0), \
			"Pass either keyword-based data or comma-separated data."

		if len(data) > 0:
			use_data = data
		elif len(kwdata) > 0:
			use_data = kwdata
		# elif _many_iter_data is not None:
		# 	use_data =
		else:
			use_data = []

		time_start = time.time()
		n_updated_records = None
		status = None

		if self.logfile is not None:
			self.logfile.write(">>> {} {} {} START UPDATE\n{}\ndata={}\nkwdata={}\n\n".format(
				self.cursor_id, self.last_update_id, time_start, sql, data, kwdata))

		# print "\n*** UPDATE QUERY:", sql, "\n"

		try:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE..."
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE..."
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE..."

			# if _many_iter_data is not None:
			# 	self._psql_unnamed_cur.executemany(sql, _many_iter_data)
			# else:
			self._psql_unnamed_cur.execute(sql, use_data)

		except (KeyboardInterrupt, SystemExit) as e:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE INTERRUPT/EXIT"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE INTERRUPT/EXIT"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE INTERRUPT/EXIT"
			self.connection._psql_conn.cancel()  # FIXME: Make it general
			sys.exit(-1)

		except QueryCanceledError:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE CANCEL"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE CANCEL"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE CANCEL"
			debug("QueryCanceledError")
			status = "QueryCanceledError"
			raise TimeLimitElapsed

		except OperationalError as e:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE OP"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE OP"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE OP"
			self.connection.reconnect()
			self.create_cursors()
			kwdata["___n_retrials"] = n_retrials + 1
			self.sql_update(sql, *data, **kwdata)

		except Exception as e:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> EXCEPTION"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> EXCEPTION"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> EXCEPTION"
			print
			print '=' * 120
			print "Exception occured while executing query:\n{}".format(self._psql_unnamed_cur.query)
			print '=' * 120
			traceback.print_exc(file=sys.stdout)
			if hasattr(e, "pgerror"):
				print e.pgerror
				status = "pgcode:{}".format(e.pgcode)
			else:
				status = "exception"

			if self.logfile:
				for x in self._update_queries:
					print x

			raise e

		else:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE OK"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE OK"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UDATE OK"

			status = "OK"
			if self.verbose:
				print self._psql_unnamed_cur.query
				print self._psql_unnamed_cur.statusmessage
			n_updated_records = self._psql_unnamed_cur.rowcount
			return n_updated_records

		finally:
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UPDATE FINALLY"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UPDATE FINALLY"
			# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> UPDATE FINALLY"
			# print ("%"*50) + "%%%%%%%%" + ("%"*49)
			# self._psql_unnamed_cur.close()
			time_end = time.time()

			self.last_update_query = self._psql_unnamed_cur.query

			# if self.logfile:
			# 	self._update_queries.append((time_start, time_end, status, sql, data, kwdata))

			if self.logfile is not None:
				self.logfile.write(">>> {} {} {} END UPDATE\n{} - {} record(s) updated\n\n".format(
					self.cursor_id, self.last_update_id, time_end, status, n_updated_records))
				self.logfile.flush()

			if self.sqlfile is not None:
				self.sqlfile.write("-- {} update\n".format(self.last_update_id))
				self.sqlfile.write(self._psql_unnamed_cur.query + ";")
				self.sqlfile.write("\n\n")



