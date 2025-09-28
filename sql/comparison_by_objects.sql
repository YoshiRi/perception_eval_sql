-- base と comp の GT行を (t4dataset_id, frame_index, uuid) で比較
WITH base_gt AS (
  SELECT
    t4dataset_id,
    frame_index,
    uuid AS gt_uuid,
    COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base
  FROM view_eval_flat
  WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND ${topic_label_id_filter:raw} AND ${visibility_filter:raw} AND ( dist_h < $max_eval_range )
  GROUP BY 1,2,3
),
comp_gt AS (
  SELECT
    t4dataset_id,
    frame_index,
    uuid AS gt_uuid,
    COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp
  FROM view_eval_flat_comp
  WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL AND ${compared_topic_label_id_filter:raw} AND ${visibility_filter:raw} AND ( dist_h < $max_eval_range )
  GROUP BY 1,2,3
),
joined AS (
  SELECT
    COALESCE(b.t4dataset_id, c.t4dataset_id) AS t4dataset_id,
    COALESCE(b.frame_index, c.frame_index)   AS frame_index,
    COALESCE(b.gt_uuid,      c.gt_uuid)      AS gt_uuid,
    COALESCE(b.tp_base, FALSE) AS tp_base,
    COALESCE(c.tp_comp, FALSE) AS tp_comp
  FROM base_gt b
  FULL OUTER JOIN comp_gt c
    ON b.t4dataset_id = c.t4dataset_id
   AND b.frame_index   = c.frame_index
   AND b.gt_uuid       = c.gt_uuid
)
SELECT
  t4dataset_id,
  CAST(COUNT(*) FILTER (WHERE TRUE)                          AS DOUBLE) AS total_gt,
  CAST(COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp)       AS DOUBLE) AS improved_cnt,   -- FN→TP
  CAST(COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp)       AS DOUBLE) AS degraded_cnt,   -- TP→FN
  CAST(COUNT(*) FILTER (WHERE tp_base AND tp_comp)           AS DOUBLE) AS both_tp_cnt,    -- TP維持
  CAST(COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp)   AS DOUBLE) AS both_fn_cnt,    -- 両方検出不可
  CAST(SUM( (CASE WHEN tp_comp THEN 1 ELSE 0 END)
           - (CASE WHEN tp_base THEN 1 ELSE 0 END))          AS DOUBLE) AS net_tp_delta
FROM joined
GROUP BY 1
ORDER BY improved_cnt DESC;
