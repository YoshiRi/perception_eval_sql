CREATE OR REPLACE VIEW view_tpr_fpr_by_class_dist_topic AS
WITH stats AS (
  SELECT
    t4dataset_id,
    topic_name,
    label,
    distance_bin,
    COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) AS gt_total,
    COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS tp_gt,
    COUNT(*) FILTER (WHERE source='EST' AND status IN ('TP','FP')) AS est_total,
    COUNT(*) FILTER (WHERE source='EST' AND status='FP') AS fp_est
  FROM view_eval_flat
  GROUP BY t4dataset_id, topic_name, label, distance_bin
)
SELECT
  *,
  CASE 
    WHEN gt_total > 0 THEN CAST(tp_gt AS DOUBLE) / gt_total
    ELSE NULL
  END AS tpr,
  CASE 
    WHEN est_total > 0 THEN CAST(fp_est AS DOUBLE) / est_total
    ELSE NULL
  END AS fpr
FROM stats;
