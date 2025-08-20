WITH base AS (
  SELECT
    distance_bin,
    CASE 
      WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total)
      ELSE 0
    END AS tpr,
    CASE 
      WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total)
      ELSE 0
    END AS fpr
  FROM view_tpr_fpr_by_class_dist_topic
  WHERE ${topic_label_id_filter:singlequate}
  GROUP BY distance_bin
),
comp AS (
  SELECT
    distance_bin,
    CASE 
      WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total)
      ELSE 0
    END AS tpr_comp,
    CASE 
      WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total)
      ELSE 0
    END AS fpr_comp
  FROM view_tpr_fpr_by_class_dist_topic_c
  WHERE ${topic_label_id_filter:singlequate}
  GROUP BY distance_bin
)
SELECT
  b.distance_bin,
  b.tpr,
  b.fpr,
  c.tpr_comp,
  c.fpr_comp,
  (c.tpr_comp - b.tpr) AS tpr_diff,
  (c.fpr_comp - b.fpr) AS fpr_diff
FROM base b
JOIN comp c USING (distance_bin)
ORDER BY CAST(REPLACE(SPLIT_PART(b.distance_bin, ',', 1), '[', ' ') AS INTEGER);
