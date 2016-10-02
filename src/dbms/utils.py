import inspect
import os
import sys
from itertools import chain

import glob2 as glob2

from src.dbms.db import DBCursor, DBConnection
from src.dbms.dbms_settings import data_dbms_settings, exp_dbms_settings



def split_schema_table_names(table_name):
    x = table_name.split(".")
    if len(x) >= 2:
        schema_name, table_name = x
    else:
        schema_name, table_name = "public", x[0]
    return schema_name, table_name




def insert_many(db, table_name, rows_iter, schema_name=None, commit=False):
    if schema_name is not None and schema_name != "public":
        table_name = "{}.{}".format(schema_name, table_name)

    cols = None
    rows = []
    tot_inserted = 0

    def insert_():
        print "Inserting {} rows...".format(len(rows))
        sql = "INSERT INTO {T} ({cols}) VALUES ({data})".format(
            T=table_name,
            cols=",".join(cols),
            data="),(".join(db.mogrify(",".join("%({})s".format(c) for c in cols), r) for r in rows),
        )
        n = db.sql_update(sql)
        if commit:
            db.commit()
        return n

    for row in rows_iter:
        if cols is None: cols = row.keys()

        rows.append(row)

        if len(rows) < 1000: continue

        tot_inserted += insert_()
        rows = []

    if len(rows) > 0:
        tot_inserted += insert_()

    return tot_inserted


def insert(db, _table_name, _schema_name=None, _return=False, **kwargs):
    if _return:
        return insert_get(db, _table_name, _schema_name=_schema_name, **kwargs)

    if _schema_name is not None and _schema_name != "public":
        _table_name = "{}.{}".format(_schema_name, _table_name)

    insert_cols = kwargs.keys()

    db.sql_update(
        "INSERT INTO {T} ({inserts}) VALUES ({formats})".format(
            T=_table_name,
            inserts=",".join(insert_cols),
            formats=",".join(["%s"] * len(insert_cols))
        ),
        **kwargs)


def insert_get(db, _table_name, _schema_name=None, **kwargs):
    if _schema_name is not None and _schema_name != "public":
        _table_name = "{}.{}".format(_schema_name, _table_name)

    insert_cols = kwargs.keys()

    return db.sql_query(
        "INSERT INTO {T} ({inserts}) VALUES ({formats}) RETURNING *".format(
            T=_table_name,
            inserts=",".join(insert_cols),
            formats=",".join("%({})s".format(c) for c in insert_cols),
        ),
        **kwargs).next()


def update(db, _table_name, _schema_name=None, _where_conds=None, _commit=False, **kwargs):
    if _schema_name is not None and _schema_name!="public":
        _table_name = "{}.{}".format(_schema_name, _table_name)

    if _where_conds is not None:
        where_conds = ("{w} = %({w})s".format(w=where_col) for where_col in _where_conds)
    else:
        where_conds = None

    updates = ("{x} = %({x})s".format(x=k) for k, v in kwargs.iteritems())

    sql = (
        "UPDATE {T} SET "
        "  {updates} "
        "  {where} "
    ).format(
        T=_table_name,
        where=("WHERE " + " AND ".join(where_conds)) if where_conds else "",
        updates=",".join(updates))

    db.sql_update(sql, **dict(kwargs, **_where_conds))

    if _commit:
        db.commit()


def upsert(db, _table_name, _where_cols=None, _schema_name=None, _return=False, **kwargs):
    """
    First, it tries to update the rows that satisfy the _where_cols conditions, using kwargs for the new values.
    If nothing satisfies the _where_cols conditions, then it will insert a new row that will, including the
    remaining data from kwargs.
    """
    if _schema_name is not None and _schema_name != "public":
        _table_name = "{}.{}".format(_schema_name, _table_name)

    if _where_cols is not None:
        where_conds = ("{w} = %({w})s".format(w=where_col) for where_col in _where_cols)
    else:
        where_conds = None

    insert_cols = []
    updates = []
    for k, v in kwargs.iteritems():
        insert_cols.append(k)
        if _where_cols is None or k not in _where_cols:
            updates.append("{x} = %({x})s".format(x=k))

    sql = (
        "WITH upsert AS ("
        "   UPDATE {T} SET "
        "       {updates} "
        "       {where} "
        "   RETURNING *) "
        "INSERT INTO {T} ({inserts}) "
        "SELECT "
        "   {selects} "
        "WHERE NOT EXISTS (SELECT * FROM upsert) "
        # "{returning}"
    ).format(
        T=_table_name,
        where=("WHERE " + " AND ".join(where_conds)) if where_conds else "",
        updates=",".join(updates),
        inserts=",".join(insert_cols),
        selects=",".join("%({})s".format(c) for c in insert_cols),
    )

    db.sql_update(sql, **kwargs)


def db_quick_connect(db_type=None, logfile=None, sqlfile=None, **kwargs):
    """
    Easily create a DB connection.

    :param db_type: The type of database to connect to. Either "data" (default) or "exp".
    :param logfile:
    :param sqlfile:
    """
    if db_type == "data":
        dbms_settings = data_dbms_settings
    elif db_type == "exp":
        dbms_settings = exp_dbms_settings
    elif db_type is None:
        dbms_settings = data_dbms_settings
    else:
        raise Exception("Database type '{}' not supported.".format(db_type))

    # Update settings from kwargs
    for key, val in kwargs.iteritems():
        dbms_settings[key] = val

    return DBCursor(DBConnection(**dbms_settings), logfile, sqlfile)


def sql_table_exists(db, schema_name, table_name):
    assert isinstance(db, DBCursor)
    return db.sql_query(
        "SELECT EXISTS (SELECT 1 "
        "FROM information_schema.tables "
        "WHERE table_schema = %s AND table_name = %s)",
        schema_name,
        table_name).next()[0]


def sql_table_column_exists(db, schema_name, table_name, column_name, data_type=None):
    assert isinstance(db, DBCursor)

    if data_type is not None:
        return db.sql_query(
            "SELECT EXISTS (SELECT 1 "
            "FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = %s AND lower(data_type) = %s)",
            schema_name,
            table_name,
            column_name,
            data_type).next()[0]
    else:
        return db.sql_query(
            "SELECT EXISTS (SELECT 1 "
            "FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = %s)",
            schema_name,
            table_name,
            column_name).next()[0]


def sql_table_column_data_type(db, table_name, column_name):
    assert isinstance(db, DBCursor)

    try:
        r = db.sql_query(
            "SELECT pg_catalog.format_type(A.atttypid, A.atttypmod) AS format_type "
            "FROM pg_catalog.pg_attribute A "
            "WHERE "
            "	A.attrelid = %s::regclass AND "
            "	A.attname = %s", table_name, column_name).next()
    except StopIteration:
        return None
    else:
        return r[0]


def sql_index_exists(db, schema_name, table_name, index_name=None, col_name=None):
    assert isinstance(db, DBCursor)
    return db.sql_query(
        "SELECT EXISTS (SELECT 1 "
        "FROM pg_index I, pg_class CI, pg_class CT, pg_namespace N, pg_attribute A "
        "WHERE I.indexrelid = CI.oid AND I.indrelid = CT.oid AND N.oid = CT.relnamespace AND N.oid = CI.relnamespace "
        "AND A.attrelid = CT.oid AND A.attnum = ANY(I.indkey) "
        "AND N.nspname = %s AND CT.relname = %s AND CI.relname LIKE %s AND A.attname LIKE %s)",
        schema_name,
        table_name,
        index_name if index_name is not None else '%',
        col_name if col_name is not None else '%',
    ).next()[0]


def sql_get_all_indexes(db, schema_name, table_name):
    assert isinstance(db, DBCursor)

    # Consider the schema name
    sql = (
        "SELECT CI.relname, A.attname, I.indisprimary "
        "FROM pg_index I, pg_class CI, pg_class CT, pg_namespace N, pg_attribute A "
        "WHERE I.indexrelid = CI.oid AND I.indrelid = CT.oid AND N.oid = CT.relnamespace AND N.oid = CI.relnamespace "
        "AND A.attrelid = CT.oid AND A.attnum = ANY(I.indkey) "
        "AND N.nspname = %s AND CT.relname = %s"
    )
    res = db.sql_query(sql, schema_name, table_name)

    res2 = []

    return [
        (r.relname, r.attname, r.indisprimary) for r in chain(res, res2)
    ]


def sql_get_all_attributes(db, table_name, schema_name=None):
    assert isinstance(db, DBCursor)

    if "." in table_name:
        assert schema_name is None
        schema_name, table_name = table_name.split(".")
    elif schema_name is None:
        schema_name = "public"

    sql = (
        "SELECT A.attname "
        "FROM pg_class CT, pg_namespace N, pg_attribute A "
        "WHERE N.oid = CT.relnamespace AND A.attrelid = CT.oid "
        "AND N.nspname = %s AND CT.relname = %s "
        "AND A.attnum >= 1 "  # This avoids System Columns: Ordinary columns are numbered from 1 up.
                              # System columns, such as oid, have (arbitrary) negative numbers
        "AND A.atttypid > 0"  # This avoids Dropped Columns: In a dropped column's pg_attribute entry,
                              # atttypid is reset to zero
    )

    return sorted(set([
        r[0].lower() for r in db.sql_query(sql, schema_name, table_name)
    ]))


def get_db_type(val):
    if type(val) is int:
        return "INT"
    elif type(val) is float:
        return "FLOAT"
    elif isinstance(val, basestring):
        return "VARCHAR"
    elif type(val) is list or type(val) is tuple:
        return get_db_type(val[0]) + "[]"
    else:
        raise Exception("Type of `{}' not recognized: {}".format(val, type(val)))


def to_db_val(val):
    if type(val) is int:
        return str(val)
    elif type(val) is float:
        return str(val)
    elif isinstance(val, basestring):
        return "'" + val + "'"
    elif type(val) is list or type(val) is tuple:
        return "ARRAY[" + ",".join(to_db_val(v) for v in val) + "]"
    else:
        raise Exception("Type of `{}' not recognized: {}".format(val, type(val)))


def sqlfile(sql_file_name):
    module_filename = inspect.getfile(sys._getframe(1))

    files = glob2.glob("{}/**/{}.sql".format(os.path.dirname(os.path.abspath(module_filename)), sql_file_name))

    if len(files) < 1:
        raise Exception("SQL file {}.sql not found!".format(sql_file_name))
    elif len(files) > 1:
        raise Exception("More than one SQL file {}.sql were found!".format(sql_file_name))

    return open(files[0]).read()
