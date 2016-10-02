from src.utils.log import log



class QuadTreePartitioning(object):
    def __init__(self, db, dataset_size, nbits, cid_type_cast,
                 data_table_name,
                 repr_table_name,
                 clust_attrs, data_attrs, max_clust_size, min_n_clusters, epsilon,
                 index_table_name,
                 indexing_attrs,
                 sr_schema,
                 obj_type=None):
        self.sr_schema = sr_schema
        self.labels_ = None
        self.N = dataset_size
        self.nbits = nbits
        self.db = db
        self.data_table_name = data_table_name
        self.repr_table_name = repr_table_name
        self.clust_attrs = clust_attrs
        self.data_attrs = data_attrs
        self.global_depth = 0
        self.max_clust_size = max_clust_size
        self.min_n_clusters = min_n_clusters
        self.epsilon = epsilon
        self.partitioned_cids = set()
        self.index_table_name = index_table_name
        self.obj_type = obj_type

        if self.epsilon is not None:
            assert self.obj_type is not None

        self.indexing_attrs = indexing_attrs

        self.clust_attrs_mask = "".join("1" if attr in self.clust_attrs else "0" for attr in self.indexing_attrs)
        log("Clust attrs mask: {}".format(self.clust_attrs_mask))

        self.partitioning_sql = None  # Will be set later, in fit()
        self.aggregating_sql = None  # Will be set later, in fit()

        self.cid_type_cast = cid_type_cast
        self.mask_type_cast = "BIT({})".format(self.nbits)


    def store_representatives(self):
        """
        Store partitioning to data table and representatives to representative table
        and set labels (partition sizes).
        """
        empirical_epsilon_max = (
            "CASE WHEN avg_{attr} > 0 THEN "
            "   (radius / avg_{attr})::float "
            "ELSE "
            "   NULL "
            "END AS emp_eps_max_{attr}"
        )
        empirical_epsilon_min = (
            "CASE WHEN avg_{attr} > 0 THEN "
            "   (radius / (avg_{attr} - radius))::float "
            "ELSE "
            "   NULL "
            "END AS emp_eps_min_{attr}"
        )

        # Store representatives
        log("Storing representative table '{repr_table}'...".format(repr_table=self.repr_table_name))
        self.db.sql_update(
            "DROP TABLE IF EXISTS {SR}.{repr_table};\n"
            "CREATE TABLE {SR}.{repr_table} AS "
            "SELECT cid, {attrs}, cid_size, radius, {emp_eps_max}, {emp_eps_min} "
            "FROM {SR}.centroids".format(
                SR=self.sr_schema,
                repr_table=self.repr_table_name,
                attrs=",".join("avg_{attr}::float as {attr}".format(attr=attr) for attr in self.data_attrs),
                emp_eps_max=",".join(empirical_epsilon_max.format(attr=attr) for attr in self.clust_attrs),
                emp_eps_min=",".join(empirical_epsilon_min.format(attr=attr) for attr in self.clust_attrs),
            ))
        log("Representative table stored.")

        # Create index on representative table
        log("Creating index on representative table...")
        self.db.sql_update(
            "CREATE INDEX ON {SR}.{repr_table} (cid)".format(
                SR=self.sr_schema,
                repr_table=self.repr_table_name))
        log("Index on representative table created.")


    def fit(self, only_representatives=False, indexing=False):
        """
        Labels is a list of cluster labels with same lenght as dataset "data".
        Each label labels[i] is a cluster index indicating which of the n_clusters clusters data[i] belongs to.
        """
        assert self.max_clust_size is not None or self.min_n_clusters is not None

        averages_sql = ", ".join(
            "AVG({attr}) AS avg_{attr}".format(attr=attr)
            for attr in self.data_attrs)

        mins_sql = ", ".join(
            "MIN({attr}) AS min_{attr}".format(attr=attr)
            for attr in self.clust_attrs)

        maxs_sql = ", ".join(
            "MAX({attr}) AS max_{attr}".format(attr=attr)
            for attr in self.clust_attrs)

        # The basic centroid table does not contain the radius yet
        centroids_basic = (
            "SELECT "
            "    cid, "
            "    COUNT(*) AS cid_size, \n"
            "    {avgs}, \n"
            "    {mins}, \n"
            "    {maxs} \n"
            "FROM {D} \n"
            "GROUP BY cid".format(
                D=self.data_table_name,
                avgs=averages_sql,
                mins=mins_sql,
                maxs=maxs_sql))

        if self.obj_type is None:
            averages_eps_val = None
        elif self.obj_type.lower() == "maximize":
            averages_eps_val = "avg_{attr} * {epsilon}"
        elif self.obj_type.lower() == "minimize":
            averages_eps_val = "avg_{attr} * ({epsilon} / (1 + {epsilon}))"
        else:
            raise Exception("Unknown objective type.")

        # This is the complete centroid table, containing radius information
        centroids_complete = (
            "SELECT "
            "	cid, \n"
            "	cid_size, \n"
            "	{averages}, \n"
            "	{radius} AS radius \n"
            "	{averages_eps} \n"
            "	{radiuses} \n"
            "FROM centroids_basic A").format(
            averages=",".join("avg_{attr}".format(attr=attr) for attr in self.data_attrs),
            radius="GREATEST({})".format(",".join(
                "A.avg_{attr} - A.min_{attr}, A.max_{attr} - A.avg_{attr}".format(attr=attr)
                for attr in self.clust_attrs)),
            # The followings are used when epsilon is set, to ensure a certain clustering quality
            averages_eps=("," + ",".join(
                ("(" + averages_eps_val + ") AS avg_eps_{attr}").format(
                    attr=attr,
                    epsilon=self.epsilon)
                for attr in self.clust_attrs)) if self.epsilon is not None else "",
            radiuses=("," + ",".join(
                "GREATEST(A.avg_{attr} - A.min_{attr}, A.max_{attr} - A.avg_{attr}) AS radius_{attr}".format(attr=attr)
                for attr in self.clust_attrs)) if self.epsilon is not None else "")

        self.db.sql_update("DROP MATERIALIZED VIEW IF EXISTS {SR}.centroids".format(SR=self.sr_schema))

        # Create the materialized view that stores the current centroids
        self.db.sql_update(
            "CREATE MATERIALIZED VIEW {SR}.centroids AS \n"
            "WITH centroids_basic AS (\n"
            "{centroids_basic} \n"
            ") \n"
            "{centroids_complete} \n"
            "WITH NO DATA".format(
                SR=self.sr_schema,
                centroids_basic=centroids_basic,
                centroids_complete=centroids_complete))

        self.aggregating_sql = "REFRESH MATERIALIZED VIEW {SR}.centroids".format(SR=self.sr_schema)

        # Define partitioning query: keep partitioning only groups that violate the size and radius conditions
        neg_size_condition = "A.cid_size > {}".format(self.max_clust_size)

        # NOTE: This condition is more fine grained that saying MAX(radius_attr) <= MIN(agv_aggr)
        # and it can cause less partitioning, although it still satisfies the approximation bounds.
        # So it's preferable.
        neg_radius_condition = "FALSE" if self.epsilon is None else " OR ".join(
            "A.radius_{attr} > A.avg_eps_{attr}".format(attr=attr) for attr in self.clust_attrs)

        self.partitioning_sql = (
            "UPDATE {D} D SET cid = ("
            # NOTE: THIS PARTITION INDEX SCHEME DOESN'T SUPPORT THE INDEX
            "(0::BIT({diff_bits}) || ({internal_cid})::BIT({k})) | (D.cid::BIT({nbits}) << {k})"
            ")::{cid_type_cast} \n"
            "FROM {SR}.centroids A \n"
            "WHERE ({neg_size_condition} OR {neg_radius_condition}) \n"
            "AND D.cid = A.cid"
            "".format(
                SR=self.sr_schema,
                D=self.data_table_name,
                internal_cid=" || ".join(
                    "(CASE"
                    "   WHEN D.{attr} IS NULL OR A.avg_{attr} IS NULL OR D.{attr} = A.avg_{attr} THEN "
                    "       round(random())::int::bit(1) "
                    "   ELSE "
                    "       (D.{attr} < A.avg_{attr})::int::bit(1) "
                    "END)".format(attr=attr)
                    for attr in self.clust_attrs),
                k=len(self.clust_attrs),
                nbits=self.nbits,
                diff_bits=self.nbits - len(self.clust_attrs),
                cid_type_cast=self.cid_type_cast,
                mask_type_cast=self.mask_type_cast,
                neg_size_condition=neg_size_condition,
                neg_radius_condition=neg_radius_condition))

        ##################################################################################
        # RUN PARTITIONING PROCESS
        ##################################################################################
        if not only_representatives:
            keep = True
            tree_level = 0
            while keep:
                keep = self.partition(tree_level, indexing) > 0
                tree_level += 1
        else:
            log("Only computing the representatives...")
            self.db.sql_update(self.aggregating_sql)
            log("Representatives computed.")

        # Store partitioning to data table
        self.store_representatives()

        # Clean up
        self.db.sql_update("DROP MATERIALIZED VIEW {SR}.centroids".format(SR=self.sr_schema))


    def partition(self, tree_level, indexing):
        log("aggregating at tree level {}...".format(tree_level))
        self.db.sql_update(self.aggregating_sql)

        if indexing:
            level_index_table_name = "{}_l{}".format(self.index_table_name, tree_level)

            # Store the aggregated meta info in separate tables, one for each tree level
            log("storing indexing level in table '{}'...".format(level_index_table_name))
            self.db.sql_update(
                "DROP TABLE IF EXISTS {SR}.{level_index_table_name};"
                "CREATE TABLE {SR}.{level_index_table_name} AS "
                "SELECT * FROM {SR}.centroids;"
                "CREATE INDEX ON {SR}.{level_index_table_name} USING btree (cid)".format(
                    SR=self.sr_schema,
                    level_index_table_name=level_index_table_name))

        log("analyzing data table...")
        self.db.sql_update("ANALYZE {D}".format(D=self.data_table_name))

        # Partition only clusters whose size is above the threshold
        log("partitioning at tree level {}...".format(tree_level))
        print
        print self.partitioning_sql.format(tree_level=tree_level)
        new_cids_n = self.db.sql_update(self.partitioning_sql.format(tree_level=tree_level))

        log("new partitions: {}...".format(new_cids_n))

        self.db.commit()

        return new_cids_n
