-- 距離ビン: 必要なら値は後で差し替え可
CREATE OR REPLACE VIEW view__binned_src AS
WITH base AS (
  SELECT
    t4dataset_id,
    topic_name,
    label,
    source,   -- 'GT' or 'EST'
    status,   -- 'TP' / 'FN' / 'FP'
    sqrt(CAST(x AS DOUBLE)*CAST(x AS DOUBLE) + CAST(y AS DOUBLE)*CAST(y AS DOUBLE)) AS dist_h
  FROM read_parquet('/srv/data/analytics/x2_gen2_komatsu_sample_current.parquet')
  WHERE x IS NOT NULL AND y IS NOT NULL
),
bins AS (
  SELECT * FROM (
    VALUES
      (0.0,   10.0,  '[0,10)',    1),
      (10.0,  20.0,  '[10,20)',   2),
      (20.0,  30.0,  '[20,30)',   3),
      (30.0,  40.0,  '[30,40)',   4),
      (40.0,  60.0,  '[40,60)',   5),
      (60.0,  80.0,  '[60,80)',   6),
      (80.0,  100.0, '[80,100)',  7),
      (100.0, 150.0, '[100,150)', 8),
      (150.0, 1e12,  '[150,inf)', 9)
  ) AS t(bin_start, bin_end, distance_bin, bin_idx)
)
SELECT
  bse.t4dataset_id,
  bse.topic_name,
  bse.label,
  bse.source,
  bse.status,
  b.distance_bin,
  b.bin_idx
FROM base bse
JOIN bins b
  ON bse.dist_h >= b.bin_start AND bse.dist_h < b.bin_end;
