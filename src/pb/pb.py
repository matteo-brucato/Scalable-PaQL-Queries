#!/usr/bin/env python
import glob
import logging
from argparse import ArgumentParser

import os
import subprocess
import sys
import signal

import time
from sshtunnel import SSHTunnelForwarder

from src.config import read_config
from src.utils.log import log, set_logging_level
from string import Template

PB_HOME = os.getenv("PB_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DB_LOGFILE_NAME = "postgres-server.log"


def tail_experiment(exp):
    from src.dbms.utils import db_quick_connect
    expdb = db_quick_connect("exp")

    stdout = open(exp.stdout, "rb")

    printed_until = 0
    while True:
        stdout.seek(printed_until)

        for line in stdout:
            print line,  # do not print newline twice
        printed_until = stdout.tell()

        # Stop if experiment is not running
        check = list(expdb.sql_query(
            "SELECT FROM exp WHERE status = 'running' AND id = %s",
            exp.id))
        if len(check) < 1:
            break

        time.sleep(0.5)

    # Print last lines from experiment (if any)
    for line in stdout:
        print line

    stdout.close()


def get_next_scheduled_experiment():
    from src.dbms.utils import db_quick_connect
    expdb = db_quick_connect("exp")
    exp = list(expdb.sql_query("SELECT * FROM exp WHERE status = 'scheduled' ORDER BY addtime, id LIMIT 1"))
    expdb.close()
    if len(exp) >= 1:
        return exp[0]
    else:
        return None


def get_current_running_experiment():
    from src.dbms.utils import db_quick_connect
    expdb = db_quick_connect("exp")
    exp = list(expdb.sql_query("SELECT * FROM exp WHERE status = 'running' ORDER BY addtime, id LIMIT 1"))
    expdb.close()
    if len(exp) >= 1:
        return exp[0]
    else:
        return None


def set_experiment_as_running(exp):
    from src.dbms.utils import db_quick_connect
    expdb = db_quick_connect("exp")
    expdb.sql_update("UPDATE exp SET status = 'running', starttime = NOW() WHERE id = %s", exp.id)
    expdb.commit()
    expdb.close()


def proc_dump(args, port):
    from src.dbms.utils import db_quick_connect
    from src.experiments.settings import datasets_directory_path

    dbname, schemaname, tablename = args

    expdb = db_quick_connect("exp")
    r = list(expdb.sql_query(
        "SELECT * FROM datasets WHERE "
        "db_name = %s AND schema_name = %s AND table_name = %s",
        dbname, schemaname, tablename))

    if len(r) < 1:
        raise Exception("Dataset table {}.{} not registered for database {}".format(
            schemaname, tablename, dbname))
    elif len(r) > 1:
        raise Exception("Ambiguous table {}.{} in database {}. More than 1 entry.".format(
            schemaname, tablename, dbname))

    outfilename = schemaname + "." + tablename + ".sql"
    dump_rel_path = os.path.join(
        datasets_directory_path,
        r[0].dump_rel_path,
        outfilename)

    newargs = [ "pg_dump"] + ([ "-p", port ] if port else []) + [
        "-f", dump_rel_path,
        "-F", "c",  # "Custom" format for pg_dump,
        "-n", schemaname,
        "-O",  # Do not output ownership commands
        "-t", tablename,
        "-v",
        dbname
    ]

    print " ".join(newargs)

    return newargs


def proc_restore(args, port):
    from src.experiments.settings import datasets_directory_path
    from src.dbms.utils import db_quick_connect
    from src.dbms.dbms_settings import data_dbms_settings, exp_dbms_settings

    from_dbname, schemaname, tablename = args

    to_dbname = data_dbms_settings["dbname"]

    expdb = db_quick_connect("exp")
    datadb = db_quick_connect("data")

    r = list(expdb.sql_query(
        "SELECT * FROM datasets WHERE "
        "db_name = %s AND schema_name = %s AND table_name = %s",
        to_dbname, schemaname, tablename))

    if len(r) >= 1:
        raise Exception("Table {}.{} in database {} already exists.".format(
            schemaname, tablename, to_dbname))

    outfilename = schemaname + "." + tablename + ".sql"
    dump_rel_path = os.path.join(
        datasets_directory_path,
        "..",
        from_dbname,
        "dumps",  # NOTE: this assumes that folder "dumps" is always used
        outfilename)

    print datasets_directory_path
    print os.path.join(datasets_directory_path, "dump")

    newargs = [ "pg_restore"] + ([ "-p", port ] if port else []) + [
        "-F", "c",  # "Custom" format for pg_dump
        "-c",  # Clean: clean (drop) database objects before recreating them
        "-n", schemaname,
        "-O",  # Do not output ownership commands
        "-t", tablename,
        "-v",
        "-d", to_dbname,
        dump_rel_path
    ]

    def callback():
        # Execute this after pg_restore has completed
        table_size = datadb.sql_query(
            "SELECT COUNT(*) FROM {}.{}".format(schemaname, tablename)).next()[0]

        expdb.sql_update(
            "INSERT INTO datasets (db_name, schema_name, table_name, table_size, dump_rel_path) "
            "VALUES (%s, %s, %s, %s, %s)",
            to_dbname, schemaname, tablename, table_size, "dumps")
        expdb.commit()

    print " ".join(newargs)

    return newargs, callback



def prepare_sql(args, folder=""):
    if len(args) > 0:
        sql_op_name = args[0].replace(".sql", "") + ".sql"
        substitute_dict = { "val{}".format(i + 1): v for i, v in enumerate(args[1:]) }
        sql_filename = os.path.join(folder, sql_op_name)
        if os.path.isfile(sql_filename):
            sql_template = Template(open(sql_filename).read())
            sql = sql_template.substitute(substitute_dict)
        else:
            sql = " ".join(args)
        return sql
    else:
        return None



def source(script, update=True, clean=True):
    """
    Source variables from a shell script
    import them in the environment (if update==True)
    and report only the script variables (if clean==True)
    """

    global environ
    if clean:
        environ_back = dict(environ)
        environ.clear()

    pipe = subprocess.Popen(". %s; env" % script, stdout=subprocess.PIPE, shell=True)
    data = pipe.communicate()[0]

    env = dict((line.split("=", 1) for line in data.splitlines()))
    print env

    if clean:
        # remove unwanted minimal vars
        env.pop('LINES', None)
        env.pop('COLUMNS', None)
        environ = dict(environ_back)

    if update:
        environ.update(env)

    return env



def print_help():
    print "Package Builder"
    print
    print "pb help              ", "\t", "Print this help message"

    print "pb                   ", "\t", "Run Python shell with PB libraries"
    print "pb set <config>      ", "\t", "Set <config>.cfg as current configuration file"
    print "pb daemon            ", "\t", "Start the PB daemon for the experiments"
    print "pb shell             ", "\t", "Run Python shell with PB libraries"
    print "pb db                ", "\t", "Go to Database shell (data db)"
    print "pb db data           ", "\t", "Go to Database shell (data db)"
    print "pb db exp            ", "\t", "Go to Database shell (experiments db)"
    print "pb db top            ", "\t", "Show current queries running in database"
    print "pb db log            ", "\t", "Tail database logs"
    print "pb db start          ", "\t", "Start database server"
    print "pb db stop           ", "\t", "Stop database server"
    print "pb db restart        ", "\t", "Restart database server"
    print "pb db <file>         ", "\t", "Run SQL file <file> in database shell"
    print "pb db <sql>          ", "\t", "Run SQL query <sql> in database shell"
    print "pb db dump           ", "\t", "Dump datasets"
    print "pb db restore        ", "\t", "Restore (load) (previously dumped) datasets"
    print "pb db kill <pid>     ", "\t", "Kill running query process <pid>"
    print "pb db killall        ", "\t", "Kill all running query processes"
    print "pb exp <name>        ", "\t", "Schedule the experimenent <name>"
    print "pb exp next          ", "\t", "Run next (previously) scheduled experiment"
    print "pb ls                ", "\t", "See scheduled experiments"
    print "pb top               ", "\t", "See running experiments"
    print "pb tail              ", "\t", "Tail currently running experiment"
    print "pb exprun <name>     ", "\t", "Run right away (without scheduler) the experimenent <name>"
    print "pb bench <name>      ", "\t", "Run the benchmark (set of experiments) <name>"
    print "pb <file>.py         ", "\t", "Run the Python script <file>.py"



def main():
    set_logging_level(logging.INFO)

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("act", nargs="?", default=None)
    args, other_args = parser.parse_known_args()

    act = args.act

    callback = None
    sshtunnel = None
    shell = False
    args = []

    if act is None or act == "shell":
        # With no arguments, simply run ipython with PB environment
        os.putenv("PYTHONSTARTUP", "src/utils/pb_startup_python_interpreter.py")
        args = [ "ipython" ]

    elif act == "help":
        print_help()
        exit(0)

    elif act == "set":
        config_name = other_args[0]
        if config_name.endswith(".cfg"): config_name = config_name[:-4]
        if not os.path.isfile(os.path.join(PB_HOME, "{}.cfg".format(config_name))):
            raise Exception("Config file `{}.cfg' does not exist in PB_HOME.".format(config_name))

        print "Setting config {}.cfg.".format(config_name)
        os.putenv("PB_CONF", config_name)
        os.system("/usr/bin/env bash")
        exit()

    elif act == "daemon":
        from src.pb import pb_daemon
        pb_daemon.main()

    elif act == "db":
        config = read_config()
        from src.dbms.dbms_settings import data_dbms_settings, exp_dbms_settings
        port = data_dbms_settings["port"] if data_dbms_settings["port"] != "" else "5432"
        host = data_dbms_settings["host"]
        orig_port = port
        db_data_folder = config.get("Folders", "dbms_folder")

        # If not localhost, perform SSH tunneling
        if host != "localhost" and host != "127.0.0.1" and host != "":
            sshtunnel = SSHTunnelForwarder((host, 22), remote_bind_address=("127.0.0.1", int(port)))
            sshtunnel.start()
            host = "localhost"
            port = str(sshtunnel.local_bind_port)

        if len(other_args) > 0:
            if other_args[0] == "log":
                base_logfile = "{}/{}".format(db_data_folder, DB_LOGFILE_NAME)
                other_logfiles = glob.glob("{}/pg_log/*".format(db_data_folder))
                log("Tailing log files:\n{}".format("\n".join([ base_logfile ] + other_logfiles)))
                args = [ "tail", "-n", "50", "-f", base_logfile ] + other_logfiles

            elif other_args[0] == "start":
                args = [ "pg_ctl", "-o", "-F", "-p " + orig_port, "-D", db_data_folder, "-l",
                         "{}/{}".format(db_data_folder, DB_LOGFILE_NAME), "start"]

            elif other_args[0] == "restart":
                opts = "-F" + \
                       (" -p " + port if port else "") + \
                       (" -h " + host if host else "")
                args = [ "pg_ctl", "-o", opts, "-D", db_data_folder, "-m", "fast", "restart" ]

            elif other_args[0] == "stop":
                args = [ "pg_ctl", "-D", db_data_folder, "-m", "fast", "stop" ]

            elif other_args[0] == data_dbms_settings["dbname"] or other_args[0] == "data":
                sql = prepare_sql(other_args[1:])
                if sql:
                    # Execute SQL on the db
                    args = [ "psql" ] + \
                           ([ "-p", port ] if port else []) + \
                           ([ "-h", host ] if host else []) + \
                           [ "-c", sql, data_dbms_settings["dbname"] ]
                else:
                    # Just connect to the db
                    args = [ "psql"] + \
                           ([ "-p", port ] if port else []) + \
                           ([ "-h", host ] if host else []) + \
                           [ data_dbms_settings["dbname"] ]

            elif other_args[0] == exp_dbms_settings["dbname"] or other_args[0] == "exp":
                sql = prepare_sql(other_args[1:])
                if sql:
                    # Execute SQL on the db
                    args = [ "psql" ] + \
                           ([ "-p", port ] if port else []) + \
                           ([ "-h", host ] if host else []) + \
                           [ "-c", sql, exp_dbms_settings["dbname"] ]
                else:
                    # Just connect to the db
                    args = [ "psql"] + \
                           ([ "-p", port ] if port else []) + \
                           ([ "-h", host ] if host else []) + \
                           [ exp_dbms_settings["dbname"] ]

            elif other_args[0] == "dump":
                args = proc_dump(other_args[1:], port)

            elif other_args[0] == "restore":
                args, callback = proc_restore(other_args[1:], port)

            else:
                # In this case, the command corresponds to a SQL query in the sql/ folder
                sql = prepare_sql(other_args + [ data_dbms_settings["dbname"], exp_dbms_settings["dbname"] ],
                                  folder=os.path.join(PB_HOME, "sql"))
                args = [ "psql" ] + \
                       ([ "-p", port ] if port else []) + \
                       ([ "-h", host ] if host else []) + \
                       [ "-c", sql, "postgres" ]

        else:
            args = [ "psql" ] + \
                   ([ "-p", port ] if port else []) + \
                   ([ "-h", host ] if host else []) + \
                   [ data_dbms_settings["dbname"] ]

    elif act == "ls":
        # See scheduled experiment
        sql = "SELECT id, addtime, exp_dir AS exp, exp_file, args " \
              "FROM exp " \
              "WHERE status = 'scheduled' " \
              "ORDER BY addtime, id "
        args = [ "pb", "db", "exp", sql ] #+ other_args

    elif act == "top":
        # See running experiment
        sql = "SELECT id, NOW() - starttime AS time_elapsed, exp_dir AS exp, exp_file, args " \
              "FROM exp " \
              "WHERE status = 'running' " \
              "ORDER BY starttime, id "
        args = [ "pb", "db", "exp", sql ] #+ other_args

    elif act == "tail":
        exp = get_current_running_experiment()
        # if exp is None:
        # 	args = [ "echo", "No running processes." ]
        # else:
        while True:
            if exp is not None:
                tail_experiment(exp)
            time.sleep(0.5)
            exp = get_current_running_experiment()

    elif act == "exp":
        if other_args[0] == "next":
            assert len(other_args) == 1
            # Run next scheduled experiment
            exp = get_next_scheduled_experiment()
            print exp.args
            args = [ "pb", "exprun", exp.exp_dir, exp.exp_file ] + exp.args.split(" ") + [
                ("--expdb", exp.expdb) if exp.expdb is not None else "",
                "--poolsize", str(exp.poolsize),
                "--repetitions", str(exp.repetitions),
                "--logging_level", str(exp.logging_level),
                "--verbose" if exp.set_verbose else "" ]
            # print args
            set_experiment_as_running(exp)
        else:
            # Schedule experiment
            args = [ "/usr/bin/env", "python", "-m", "src.experiments.run_experiments" ] + other_args

    elif act == "exprun":
        # Run experiment now
        args = [ "/usr/bin/env", "python", "-m", "src.experiments.run_experiments", "--run-now" ] + other_args

    elif act == "bench":
        # Run benchmark
        benchmark_name = other_args[0]
        args = [ "/usr/bin/env", "python", "-m", "benchmark.{}.run".format(benchmark_name) ] + other_args[1:]

    elif act.endswith(".py") and os.path.isfile(act):
        # Run Python script
        args = [ "/usr/bin/env", "python", "-m", act.replace("/", ".").replace(".py", "") ] + other_args

    else:
        raise Exception("Action '{}' not supported.".format(act))

    print "RUNNING:"
    print " ".join(args)

    if len(args):

        p = subprocess.Popen(args, env=os.environ, shell=shell)

        while p.poll() is None:
            try:
                p.wait()

            except (KeyboardInterrupt, SystemExit) as e:
                print (KeyboardInterrupt, SystemExit)
                p.send_signal(signal.SIGINT)
                # p.wait()
                if sshtunnel is not None:
                    sshtunnel.close()
                    sshtunnel = None
                sys.exit(-1)

            except Exception as e:
                print Exception
                p.send_signal(signal.SIGTERM)
                # p.wait()
                if sshtunnel is not None:
                    sshtunnel.close()
                    sshtunnel = None
                raise e

        if callback is not None:
            callback()

        if sshtunnel is not None:
            sshtunnel.close()
            sshtunnel = None




if __name__ == "__main__":
    main()
