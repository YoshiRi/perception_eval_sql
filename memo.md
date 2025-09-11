# memo

This file memorize small tips and sqls for visualization in Grafana.

## often encountered error

### empty string in query

Somehow grafana can not recognize `''` as empty string and it seems it ignore `''` in query.

### docker plugin load error

I've encountered following error when I tried to load duckdb plugin in grafana docker container.

```dockerfile
# --- Dockerfileで直接ダウンロードすると失敗するcaseではホストのzip_pluginフォルダに事前に置いておき、COPYでコンテナにコピーしてから展開する
# Copy pre-downloaded DuckDB plugin zip file to the container
# Run in Host: 
# mkdir -p zip_plugin
# curl -fL -o zip_plugin/motherduck-duckdb-datasource.zip \
#   "https://github.com/motherduckdb/grafana-duckdb-datasource/releases/download/v0.3.1/motherduck-duckdb-datasource-0.3.1.zip"
# Run in Dockerfile:
COPY zip_plugin/motherduck-duckdb-datasource.zip /tmp/plugin.zip
RUN unzip -q /tmp/plugin.zip -d /var/lib/grafana/plugins/ && \
    rm /tmp/plugin.zip
```

### grafana version issue

This board expect grafana version higher than 12.1.
It seems older grafana can not handle multiple layer variable expansion.

## TBD

## Look at the table

If you want to look at the table structure, you can use `DESCRIBE` command in SQL.

```sql
DESCRIBE view_eval_flat;
```

## set all value in grafana to achieve select all

create variable with all values and set `'__all__'` as default all value

you can use `WHERE IN ($variable)` in query

```sql
-- only write WHWERE 
WHERE  '__all__' IN (${variable:sqlstring}) OR table_column IN (${variable:sqlstring})
```

## order by distance bin

We use bin like `[0, 10), [10, 20), ...` to order by distance in `view_eval_flat` table.
It causes the ordering to be done by the first element of the bin.

```sql
ORDER BY CAST(REPLACE(SPLIT_PART(distance_bin, ',', 1), '[', ' ') AS INTEGER);
```


## filter design

For example we use `label` variable in query


```sql
WHERE
  ( '__all__' IN (${label:sqlstring}) OR label IN (${label:sqlstring}) )
```

So we use following query to filter by variables:

```sql
WHERE
(('__all' IN (${source:sqlstring})      OR source       IN (${source:sqlstring}))
 AND ('__all' IN (${label:sqlstring})       OR label        IN (${label:sqlstring}))
 AND ('__all' IN (${t4dataset_id:sqlstring}) OR t4dataset_id IN (${t4dataset_id:sqlstring}))
 AND ('__all' IN (${topic_name:sqlstring})  OR topic_name   IN (${topic_name:sqlstring})))
```

and we can set variable `***_filter` to skip writing and use it in query like this:

```sql
WHERE ${***_filter:sqlstring}
```