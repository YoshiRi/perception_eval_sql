CREATE OR REPLACE VIEW tpfp_rate_by_t4dataset_view AS
SELECT
    t4dataset_id,

    CASE 
        WHEN SUM(gt_total) > 0 
        THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total)
        ELSE 0
    END AS tpr,

    CASE 
        WHEN SUM(est_total) > 0 
        THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total)
        ELSE 0
    END AS fpr

FROM view_tpr_fpr_by_class_dist_topic
WHERE ${topic_label_id_filter:raw}
GROUP BY t4dataset_id
ORDER BY tpr DESC;
