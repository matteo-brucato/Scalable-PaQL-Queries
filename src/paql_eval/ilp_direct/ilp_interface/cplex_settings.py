import os
from cplex import Cplex
from src.experiments.settings import result_data_directory_path

CPLEX_ERROR_STREAM = None
CPLEX_LOG_STREAM = None
CPLEX_WARNING_STREAM = None
CPLEX_RESULTS_STREAM = None

# This was recommended by https://www.or-exchange.org/questions/7565/cplex-python-api-performance-overhead
CPLEX_READ_DATACHECK = 0

# Specifies an upper limit on the amount of central memory, in megabytes, that CPLEX is permitted
# to use for working memory before swapping to disk files, compressing memory, or taking other actions.
# default: 2048
# CPLEX_WORKMEM = 512  # <<== Value used in SketchRefine experiments
CPLEX_WORKMEM = 1024 * 4

# Directory for working files
CPLEX_WORKDIR = os.path.abspath(os.path.join(result_data_directory_path, "CPLEX_working_dir"))

# Sets an absolute upper limit on the size (in megabytes, uncompressed) of the branch-and-cut tree.
# If this limit is exceeded, CPLEX terminates optimization.
# default: 1e+75.
CPLEX_TREE_MEMORY_LIMIT = CPLEX_WORKMEM * 2

# MIP emphasis switch
# 0 	CPX_MIPEMPHASIS_BALANCED 	Balance optimality and feasibility; default
# 1 	CPX_MIPEMPHASIS_FEASIBILITY Emphasize feasibility over optimality
# 2 	CPX_MIPEMPHASIS_OPTIMALITY 	Emphasize optimality over feasibility
# 3 	CPX_MIPEMPHASIS_BESTBOUND 	Emphasize moving best bound
# 4 	CPX_MIPEMPHASIS_HIDDENFEAS 	Emphasize finding hidden feasible solutions
CPLEX_EMPHASIS_MIP_BALANCED    = 0
CPLEX_EMPHASIS_MIP_FEASIBILITY = 1
CPLEX_EMPHASIS_MIP_OPTIMALITY  = 2
CPLEX_EMPHASIS_MIP_BESTBOUND   = 3
CPLEX_EMPHASIS_MIP_HIDDENFEAS  = 4

# Auto MIP emphasis switch for optimization queries
CPLEX_EMPHASIS_MIP_OPTIMIZATION_QUERIES = CPLEX_EMPHASIS_MIP_OPTIMALITY

# Auto MIP emphasis switch for exploratory queries
CPLEX_EMPHASIS_MIP_EXPLORATION_QUERIES = CPLEX_EMPHASIS_MIP_FEASIBILITY

# Limits the number of presolve passes that CPLEX makes during preprocessing.
# When this parameter is set to a positive value, presolve is applied the specified number of times, or until no more
# reductions are possible.
# At the default value of -1, presolve continues only if it seems to be helping.
# When this parameter is set to zero, CPLEX does not enter its main presolve loop, but other reductions may occur,
# depending on settings of other parameters and characteristics of your model. In other words, setting this parameter to
# 0 (zero) is not equivalent to turning off the presolve switch (CPX_PARAM_PREIND, PreInd). To turn off presolve, use
# the presolve switch (CPX_PARAM_PREIND, PreInd) instead.
CPLEX_PREPROCESSING_NUMPASS = 1000

# Node storage file switch
# Used when working memory (CPX_PARAM_WORKMEM, WorkMem) has been exceeded by the size of the tree.
# If the node file parameter is set to zero when the tree memory limit is reached, optimization is terminated.
# Otherwise, a group of nodes is removed from the in-memory set as needed.
# By default, CPLEX transfers nodes to node files when the in-memory set is larger than 128 MBytes, and it keeps
# the resulting node files in compressed form in memory. At settings 2 and 3, the node files are transferred to
# disk, in uncompressed and compressed form respectively, into a directory named by the working directory
# parameter (CPX_PARAM_WORKDIR, WorkDir), and CPLEX actively manages which nodes remain in memory for processing.
CPLEX_MIP_STORAGE_FILE_DISABLED = 0
CPLEX_MIP_STORAGE_FILE_IN_MEMORY_COMPRESSED = 1
CPLEX_MIP_STORAGE_FILE_ON_DISK_UNCOMPRESSED = 2
CPLEX_MIP_STORAGE_FILE_ON_DISK_COMPRESSED = 3
# Setting
CPLEX_MIP_STORAGE_FILE = CPLEX_MIP_STORAGE_FILE_ON_DISK_COMPRESSED



def set_cplex_parameters(c):
	assert isinstance(c, Cplex)

	# Don't print anything on output streams
	# c.set_error_stream(CPLEX_ERROR_STREAM)
	# c.set_log_stream(CPLEX_LOG_STREAM)
	# c.set_warning_stream(CPLEX_WARNING_STREAM)
	# c.set_results_stream(CPLEX_RESULTS_STREAM)

	# This was recommended by https://www.or-exchange.org/questions/7565/cplex-python-api-performance-overhead
	c.parameters.read.datacheck.set(CPLEX_READ_DATACHECK)

	# Memory reduction switch (Memory Emphasis)
	# Directs CPLEX that it should conserve memory where possible. When you set this parameter to its non-default
	# value, CPLEX will choose tactics, such as data compression or disk storage, for some of the data computed by
	# the simplex, barrier, and MIP optimizers. Of course, conserving memory may impact performance in some models.
	# Also, while solution information will be available after optimization, certain computations that require a basis
	# that has been factored (for example, for the computation of the condition number Kappa) may be unavailable.
	# NOTE: YOU NEED THIS SET TO 1 TO RUN CPLEX ON YEEHA!
	c.parameters.emphasis.memory.set(1)

	# Memory available for working storage
	# Specifies an upper limit on the amount of central memory, in megabytes, that CPLEX is permitted to use for
	# working memory before swapping to disk files, compressing memory, or taking other actions.
	c.parameters.workmem.set(CPLEX_WORKMEM)

	# Directory for working files
	# Specifies the name of an existing directory into which CPLEX may store temporary working files, such as for
	# MIP node files or for out-of-core barrier files. The default is the current working directory.
	# This parameter accepts a string as its value. If you change either the API string encoding switch or the file
	# encoding switch from their default value to a multi-byte encoding where a NULL byte can occur within the encoding
	# of a character, you must take into account the issues documented in the topic Selecting an encoding in the CPLEX
	# User's Manual. Especially consider the possibility that a NULL byte occurring in the encoding of a character can
	# inadvertently signal the termination of a string, such as a filename or directory path, and thus provoke
	# surprising or incorrect results.
	# Tip: If the string designating the path to the target directory includes one or more spaces, be sure to include
	# the entire string in double quotation marks.
	c.parameters.workdir.set(CPLEX_WORKDIR)

	# Sets an absolute upper limit on the size (in megabytes, uncompressed) of the branch-and-cut tree.
	# If this limit is exceeded, CPLEX terminates optimization.
	# c.parameters.mip.limits.treememory.set(CPLEX_TREE_MEMORY_LIMIT)

	# Limits the number of presolve passes that CPLEX makes during preprocessing.
	# When this parameter is set to a positive value, presolve is applied the specified number of times,
	# or until no more reductions are possible.
	# At the default value of -1, presolve continues only if it seems to be helping.
	# When this parameter is set to zero, CPLEX does not enter its main presolve loop, but other
	# reductions may occur, depending on settings of other parameters and characteristics of your model.
	# In other words, setting this parameter to 0 (zero) is not equivalent to turning off the presolve
	# switch (CPX_PARAM_PREIND, PreInd). To turn off presolve, use the presolve switch (CPX_PARAM_PREIND,
	# PreInd) instead.
	# c.parameters.preprocessing.numpass.set(CPLEX_PREPROCESSING_NUMPASS)

	c.parameters.mip.strategy.file.set(CPLEX_MIP_STORAGE_FILE)

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# MIP emphasis switch
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# c.parameters.emphasis.mip.set(CPLEX_EMPHASIS_MIP_OPTIMIZATION_QUERIES)  # <<== This was used in the SketchRefine experiments
	# c.parameters.emphasis.mip.set(CPLEX_EMPHASIS_MIP_FEASIBILITY)
	c.parameters.emphasis.mip.set(CPLEX_EMPHASIS_MIP_BALANCED)
	# c.parameters.emphasis.mip.set(CPLEX_EMPHASIS_MIP_OPTIMALITY)

	# Feasopt Tolerance
	# Controls the amount of relaxation for the routine CPXfeasopt in the C API or for the method
	# feasOpt in the object-oriented APIs.
	# In the case of a MIP, it serves the purpose of the absolute gap for the feasOpt model in
	# Phase I (the phase to minimize relaxation).
	# Using this parameter, you can implement other stopping criteria as well. To do so, first call
	# feasOpt with the stopping criteria that you prefer; then set this parameter to the resulting
	# objective of the Phase I model; unset the other stopping criteria, and call feasOpt again.
	# Since the solution from the first call already matches this parameter, Phase I will terminate
	# immediately in this second call to feasOpt, and Phase II will start.
	# In the case of an LP, this parameter controls the lower objective limit for Phase I of feasOpt
	# and is thus relevant only when the primal optimizer is in use.
	# Values: Any nonnegative value; default: 1e-6.
	# c.parameters.feasopt.tolerance.set(0.05)
