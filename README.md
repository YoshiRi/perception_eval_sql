# Perception Evaluation with DuckDB and Grafana

This repository provides SQL queries and Grafana configuration for analysing perception system results. A custom Docker image sets up Grafana with a DuckDB data source plugin so that evaluation metrics can be visualised out of the box.

## Repository Layout

- `sql/` – SQL scripts defining views and helper queries for evaluation.
  - `view_eval_flat.sql` flattens parquet files and assigns distance bins to each record.
  - `view_tpr_fpr.sql` aggregates true/false positive rates by dataset, topic, class and distance.
  - `comparison_between_file.sql` compares metrics between two result files.
- `dashboard/` – example Grafana dashboard JSON files.
- `docker/` – provisioning files used by the Docker image (data sources and dashboard loaders).
- `Dockerfile` – builds a Grafana image with the DuckDB plugin installed and provisioning enabled.

## Getting Started

1. Build the Docker image:

   ```bash
   docker build -t perception-eval .
   ```

2. Run the container, mounting your data and SQL directories and exposing Grafana:

   ```bash
   docker run --rm -p 3000:3000 \
      -v $PWD/data:/opt/grafana/data \
      -v $PWD/sql:/opt/grafana/sql \
      -e GF_SECURITY_ADMIN_USER=admin \
      -e GF_SECURITY_ADMIN_PASSWORD=secret \
        -e GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=motherduck-duckdb-datasource\
      perception-eval
   ```
   username: `admin`, password: `secret`

3. Access Grafana at [http://localhost:3000](http://localhost:3000) and open dashboards under **Local JSON Dashboards**.

### option: when you allow access from your local network

```bash
docker run -d --name grafana-perception -p 3000:3000 \
        -v "$PWD/data:/opt/grafana/data" \
        -v "$PWD/sql:/opt/grafana/sql" \
        -v "$PWD/dashboard:/opt/grafana/dashboard" \
        -e GF_AUTH_ANONYMOUS_ENABLED=true \
        -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
        -e GF_SECURITY_ADMIN_USER=admin \
        -e GF_SECURITY_ADMIN_PASSWORD=secret \
        -e GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=motherduck-duckdb-datasource \
        perception-eval
```

## SQL Views

The SQL files define reusable views that simplify analysis. For example, `view_eval_flat.sql` computes horizontal distance for each detection and marks whether it is TP/FP/FN in fixed distance bins. `view_tpr_fpr.sql` builds on this view to calculate true positive rate (TPR) and false positive rate (FPR) for each combination of dataset, topic, class and distance bin.

## License

Specify your project's license here.

