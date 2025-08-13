CREATE OR REPLACE VIEW view_eval_flat AS
WITH base AS (
  SELECT
    *,
    sqrt(CAST(x AS DOUBLE)*CAST(x AS DOUBLE) + CAST(y AS DOUBLE)*CAST(y AS DOUBLE)) AS dist_h
  FROM parquet_scan(${target_file:sqlstring})
  WHERE x IS NOT NULL AND y IS NOT NULL
),
bins AS (
  SELECT * FROM (
    VALUES
      (0.0,   10.0,   '[0,10)',     10),
      (10.0,  20.0,   '[10,20)',    20),
      (20.0,  30.0,   '[20,30)',    30),
      (30.0,  40.0,   '[30,40)',    40),
      (40.0,  50.0,   '[40,50)',    50),
      (50.0,  60.0,   '[50,60)',    60),
      (60.0,  70.0,   '[60,70)',    70),
      (70.0,  80.0,   '[70,80)',    80),
      (80.0,  90.0,   '[80,90)',    90),
      (90.0,  100.0,  '[90,100)',  100),
      (100.0, 110.0,  '[100,110)', 110),
      (110.0, 120.0,  '[110,120)', 120),
      (120.0, 130.0,  '[120,130)', 130),
      (130.0, 140.0,  '[130,140)', 140),
      (140.0, 150.0,  '[140,150)', 150),
      (150.0, 1e12,   '[150,inf)', 160)
  ) AS t(bin_start, bin_end, distance_bin, bin_idx)
)
SELECT
  bse.*,
  b.distance_bin,
  b.bin_idx,
  (status = 'TP') AS is_tp,
  (status = 'FP') AS is_fp,
  (status = 'FN') AS is_fn
FROM base bse
JOIN bins b
  ON bse.dist_h >= b.bin_start AND bse.dist_h < b.bin_end;
