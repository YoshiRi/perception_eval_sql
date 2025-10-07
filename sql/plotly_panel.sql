-- 回転BBOX（4隅）を持つスナップショットテーブルに更新
CREATE OR REPLACE TABLE bev_bboxes AS
WITH base AS (
  SELECT
    frame_index AS ts,
    x, y, width, length, yaw,      -- yaw はラジアン
    label,
    source,
    dist_h,
    topic_name,
    t4dataset_id,
    visibility
  FROM view_eval_flat
),
param AS (
  SELECT
    *,
    width  / 2.0 AS hx,            -- half width
    length / 2.0 AS hy             -- half length
  FROM base
)
SELECT
  ts,
  x, y, width, length, yaw,
  -- 4隅（時計回り: FR, FL, BL, BR）
  ( x + ( +hx * cos(yaw) - +hy * sin(yaw) ) ) AS x1,
  ( y + ( +hx * sin(yaw) + +hy * cos(yaw) ) ) AS y1,  -- Front-Right
  ( x + ( -hx * cos(yaw) - +hy * sin(yaw) ) ) AS x2,
  ( y + ( -hx * sin(yaw) + +hy * cos(yaw) ) ) AS y2,  -- Front-Left
  ( x + ( -hx * cos(yaw) - -hy * sin(yaw) ) ) AS x3,
  ( y + ( -hx * sin(yaw) + -hy * cos(yaw) ) ) AS y3,  -- Back-Left
  ( x + ( +hx * cos(yaw) - -hy * sin(yaw) ) ) AS x4,
  ( y + ( +hx * sin(yaw) + -hy * cos(yaw) ) ) AS y4,  -- Back-Right
  label,
  source,
  dist_h,
  topic_name,
  t4dataset_id,
  visibility
FROM param;

SELECT *
FROM bev_bboxes
WHERE
  ${topic_label_id_filter:raw}
  AND (dist_h < $max_eval_range)
  AND ${visibility_filter:raw}
ORDER BY ts;
