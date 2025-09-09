# Ubuntuベースの公式Grafana（※Ubuntu版必須）
FROM grafana/grafana:10.4.2-ubuntu

# --------------- build args ---------------
ARG PLUGIN_VERSION=0.3.1
ARG TARGETARCH
# -----------------------------------------

USER root
RUN set -eux \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl unzip \
 && rm -rf /var/lib/apt/lists/*

# ディレクトリ作成
RUN mkdir -p /opt/grafana/{dashboard,data,sql} \
           /etc/grafana/provisioning/{datasources,dashboards} \
           /var/lib/grafana/plugins

# 資材コピー
COPY dashboard /opt/grafana/dashboard
COPY data      /opt/grafana/data
COPY sql       /opt/grafana/sql

# DuckDBプラグイン（ビルド済みZIPを取得して展開）
RUN set -eux; \
  curl -fL -o /tmp/duckdb-plugin.zip \
    "https://github.com/motherduckdb/grafana-duckdb-datasource/releases/download/v${PLUGIN_VERSION}/motherduck-duckdb-datasource-${PLUGIN_VERSION}.zip" \
  && mkdir -p /var/lib/grafana/plugins/motherduck-duckdb-datasource \
  && unzip -q /tmp/duckdb-plugin.zip -d /var/lib/grafana/plugins/motherduck-duckdb-datasource \
  && rm /tmp/duckdb-plugin.zip

# プロビジョニング
COPY docker/provisioning/datasources/duckdb.yml    /etc/grafana/provisioning/datasources/duckdb.yml
COPY docker/provisioning/dashboards/dashboards.yml /etc/grafana/provisioning/dashboards/dashboards.yml

# 未署名プラグイン許可（IDは "motherduck-duckdb-datasource"）
ENV GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=motherduck-duckdb-datasource

# 権限：Grafanaは uid=472, gid=0(root)
RUN chown -R 472:0 /opt/grafana /var/lib/grafana /etc/grafana

USER grafana
EXPOSE 3000
