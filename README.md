Scalable Package Builder System
===============================

This project contains the code for running experiments with the DIRECT
and SKETCHREFINE algorithms presented in the paper:

Matteo Brucato, Juan Felipe Beltran, Azza Abouzied, Alexandra Meliou:
_Scalable Package Queries in Relational Database Systems_. PVLDB 9(7): 576-587 (2016)

* Contact Author: Matteo Brucato
* Webpage: https://people.cs.umass.edu/~matteo/

## Quick Setup

1.  Run `make` from the root of this project (the folder that contains bin, etc).

2.  Install PostgreSQL and create a database that contains your input tables.

3.  Execute the SQL script "scripts/prepare\_data\_db.sql".

4.  Set up an environmental varible called PB_HOME (and export it) that points
    to the root of this project.

5.  Modify the file "cfg/example.cfg" (or create a new similar file) to reflect
    your database connection. Concentrate on \[Data DB\] (all settings) and 
    \[Folders\] "dbms\_folder". You don't need to modify the other settings.

6.  Run `bin/pb set cfg/example.cfg` or use the setting file that you have created in step 5.

7.  Create a file containing your PaQL query. Suppose this file is called "query.paql".

8.  To solve the query using DIRECT, run:
    
    `bin/pb exprun paql_eval direct -q query.paql`

9.  To solve the query using SKETCHREFINE, run:
    
    `bin/pb exprun paql_eval sketchrefine -q query.paql -a* -C .10`
    
    Where -a* means "partition the dataset on all of the query attributes"
    and -C .10 means to partition until each partition is no more than 10%
    of the input dataset size.
    
    Notice that the first time you run it, it will firstly partition the
    dataset. Then, when you re-run SKETCHREFINE with the same options, 
    the partitioning phase will not be performed again: the system is able
    to detect whether the dataset is currently partitioned in the correct
    way. You can always bypass this automatic check by using the option
    --already-partitioned.

10. Read "src/experiments/paql\_eval/sketchrefine.py" to learn the other 
    command-line options you have for SKETCHREFINE, 
    and "src/experiments/paql_eval/main.py" for the command-line options
    available for both DIRECT and SKETCHREFINE. For instance, you
    can list the exact partitioning columns you want to use, an absolute
    maximum partition size, an epsilon value for quality guarantee, time
    and memory limits, etc.
