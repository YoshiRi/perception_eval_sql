# Ubuntuベースの公式Grafana
FROM grafana/grafana:10.4.2-ubuntu

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

# ディレクトリ作成
RUN mkdir -p /opt/grafana/{dashboard,data,sql} \
           /etc/grafana/provisioning/{datasources,dashboards} \
           /var/lib/grafana/plugins

# 資材コピー
COPY dashboard /opt/grafana/dashboard
COPY data      /opt/grafana/data
COPY sql       /opt/grafana/sql

# DuckDBプラグイン導入（未署名）
RUN git clone --depth=1 https://github.com/motherduckdb/grafana-duckdb-datasource \
    /var/lib/grafana/plugins/motherduck-duckdb-datasource

# プロビジョニング
COPY docker/provisioning/datasources/duckdb.yml    /etc/grafana/provisioning/datasources/duckdb.yml
COPY docker/provisioning/dashboards/dashboards.yml /etc/grafana/provisioning/dashboards/dashboards.yml

# 未署名プラグイン許可（※パスワードはDockerfileに焼かない）
ENV GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=motherduck-duckdb-datasource

# 権限：Grafanaは uid=472, gid=0(root) を想定
# ここを 'grafana:grafana' ではなく '472:0' にするのがポイント
RUN chown -R 472:0 /opt/grafana /var/lib/grafana /etc/grafana

# 実行ユーザーを grafana に戻す
USER grafana

EXPOSE 3000
