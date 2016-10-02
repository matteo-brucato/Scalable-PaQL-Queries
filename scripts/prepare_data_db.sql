CREATE SCHEMA sketchrefine;

CREATE TABLE sketchrefine.currently_loaded_clustering(
  table_name varchar PRIMARY KEY,
  clust_table_name varchar NOT NULL, -- "cid" if "table_name" reflects the content of "clust_table_name"
  -- Settings
  clust_setting varchar NOT NULL,
  clust_algo varchar NOT NULL,
  clust_attrs varchar[] NOT NULL,
  epsilon numeric, -- Can be NULL if you don't enforce any epsilon
  -- Stats
  effective_max_clust_size integer NOT NULL,
  effective_n_clusts integer NOT NULL
);
