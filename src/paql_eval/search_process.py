import os
import sys
import time
import traceback
from multiprocessing import Pipe, Process
from signal import SIGINT

import psutil

from src.data_model.memory_representation.sequence_based import SequenceBasedPackage
from src.paql_eval.exceptions import SubprocessKilled
from src.paql_eval.search import Search
from src.utils.log import log



def get_optimal_package__process(search_class, __init__kwargs, init_kwargs, search_kwargs, parent_conn, init_only):
    assert issubclass(search_class, Search)

    # print "SUBPROCESS STARTED..."
    parent_conn[0].close()

    # Wait to receive the start from the parent process
    parent_conn[1].recv()

    # Create search class
    search = search_class(**__init__kwargs)

    # INIT
    log("Initing query for eval method: {}...".format(search_class.__name__))
    search.init(**init_kwargs)
    log("init done.")

    # Send query initialization info right away
    opt_init_info = search.get_init_info()
    print search_class
    print opt_init_info, type(opt_init_info)
    parent_conn[1].send({
        "status": "init",
        "opt_init_info": opt_init_info,
    })

    if not init_only:
        # Try to solve optimally
        try:
            log("Getting opt package using args: {}...".format(search_kwargs))
            opt_p = search.get_optimal_package(**search_kwargs)
            assert isinstance(opt_p, SequenceBasedPackage)
            opt_combo = opt_p.combination
            log("Getting package objective value...")
            opt_obj_val = opt_p.get_objective_value()
            log("Checking if package is feasible...")
            opt_is_valid = opt_p.is_valid()

            # Info about the paql_eval run to produce the package
            opt_run_info = search.get_last_run_info()

        except Exception as e:
            print "EXCEPTION:", e
            print traceback.print_exc()
            opt_run_info = search.get_last_run_info()
            parent_conn[1].send({
                "status": "exception",
                "exception": e,
                "opt_run_info": opt_run_info,
            })

        else:
            parent_conn[1].send({
                "status": "success",

                # "saerch": search,
                "opt_package_tablename": opt_p.table_name,
                # "package_in_memory": opt_p.convert_to(InMemoryPackage),

                "opt_combo": opt_combo,
                "opt_obj_val": opt_obj_val,
                "opt_is_valid": opt_is_valid,
                "opt_run_info": opt_run_info,
            })

        finally:
            search.close()
            parent_conn[1].close()
            log("done.")

    else:
        parent_conn[1].send({
            "status": "exception",
            "exception": "init_only",
            "opt_run_info": None,
        })




def check_pid_memory_usage__process(pid, mem_limit, parent_conn):
    # expdb._connection._psql_conn.close()
    parent_conn[0].close()
    # print "\n>>> CHECKING MEMORY USAGE OF SOLVER PROCESS pid =", pid

    start_time = time.time()

    i = 0
    proc = psutil.Process(pid)

    max_resident = max_virtual = None
    while True:
        # Get memory usage
        try:
            resident_memory, virtual_memory = proc.memory_info()

        except psutil.NoSuchProcess:
            break

        except psutil.AccessDenied:
            break

        max_resident = max(max_resident, resident_memory) if max_resident is not None else resident_memory
        max_virtual = max(max_virtual, virtual_memory) if max_virtual is not None else virtual_memory

        # Print every 60 * 5 rounds (about 5 minutes)
        i = (i + 1) % (60 * 5)
        if i == 1:
            print ">>> MEMORY USAGE OF SOLVER PROCESS {}: RSS={:.3f}MB, VMS={:.3f}MB; TIME ALIVE={:.0f}m".format(
                pid,
                resident_memory / float(2 ** 20),
                virtual_memory / float(2 ** 20),
                (time.time() - start_time) / 60)
            sys.stdout.flush()

        # Kill solver process, and then terminate, if it uses too much memory
        if 0 < mem_limit < resident_memory:
            print ">>> KILLING ILP SOLVER PROCESS FOR EXCESSIVE MEMORY USAGE!"
            # os.kill(pid, signal.SIGKILL)
            proc.kill()
            break

        # Terminate when you get something from parent process
        if parent_conn[1].poll():
            break

        time.sleep(1.0)

    parent_conn[1].send({ "resident": max_resident, "virtual": max_virtual })
    parent_conn[1].close()



def get_optimal_package_in_subprocess_and_monitor_memory_usage(
        search_class, __init__kwargs, init_kwargs, search_kwargs, init_only, mem_limit):
    """
    This yields two things: 1) the init info, 2) the run info (or run exception)
    """

    # Pipes to communicate with sub-processes
    solver_parent_conn, solver_child_conn = Pipe()
    # check_queue = Queue()
    checker_parent_conn, checker_child_conn = Pipe()

    # Launch the ILP solver process
    solver = Process(target=get_optimal_package__process,
                     args=(search_class,
                           __init__kwargs,
                           init_kwargs,
                           search_kwargs,
                           (solver_parent_conn, solver_child_conn),
                           init_only))
    solver.daemon = False
    solver.start()
    solver_child_conn.close()
    solver_pid = solver.pid

    # Run a process to monitor the solver's memory usage and kill it if necessary
    checker = Process(target=check_pid_memory_usage__process,
                      args=(solver_pid,
                            mem_limit,
                            (checker_parent_conn, checker_child_conn)))
    checker.daemon = False
    checker.start()
    checker_child_conn.close()

    # Let the solving process start
    solver_parent_conn.send(1)

    # Get solver results
    log("Waiting for solver process to finish...")
    try:
        # First, we get the init info
        init_result = solver_parent_conn.recv()
        yield init_result

        # Second, we get the run info
        print ">"*100
        run_result = solver_parent_conn.recv()
        print ">"*100

    # except Exception as e:
    except (KeyboardInterrupt, SystemExit) as e:
        print "=================================================================="
        print "=================================================================="
        print "============ K E Y B O A R D    I N T E R R U P T ================"
        print "=================================================================="
        print "=================================================================="
        os.kill(solver_pid, SIGINT)
        sys.exit(-1)

    except EOFError as e:
        # Could not read from queue, or something else bad happened
        # (e.g. could pickle some data to the subprocess)
        print "Pipe Connection Exception:", e.__class__.__name__
        print "Likely, the ILP solver process used too much memory and got killed."
        run_result = {
            "status": "exception",
            "exception": SubprocessKilled(),
        }

    print "SUBPROCESS RESULT STATUS:", run_result["status"]

    # Make sure the memory check process always terminates
    try:
        checker_parent_conn.send(1)
    except Exception:
        pass

    # Join with the processes
    solver.join()
    checker.join()

    # Get results from memory check process
    max_resident = max_virtual = None
    try:
        check = checker_parent_conn.recv()
    except (EOFError, IOError):
        pass
    else:
        max_resident, max_virtual = check["resident"], check["virtual"]

    # Check results of solver process
    run_result["opt_max_resident"] = max_resident
    run_result["opt_max_virtual"] = max_virtual

    # Close process connections
    solver_parent_conn.close()
    checker_parent_conn.close()

    print "YIELDING RESULTS:", run_result
    yield run_result
