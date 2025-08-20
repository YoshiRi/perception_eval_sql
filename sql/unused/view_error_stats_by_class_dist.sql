CREATE OR REPLACE VIEW view_error_stats_by_class_dist AS
SELECT
    t4dataset_id,
    topic_name,
    label,
    distance_bin,

    COUNT(*) AS tp_count,

    quantile_cont(x_error, 0.5) AS x_error_p50,
    quantile_cont(x_error, 0.9) AS x_error_p90,
    avg(x_error) AS x_error_mean,
    stddev_samp(x_error) AS x_error_std,

    quantile_cont(y_error, 0.5) AS y_error_p50,
    quantile_cont(y_error, 0.9) AS y_error_p90,
    avg(y_error) AS y_error_mean,
    stddev_samp(y_error) AS y_error_std,

    quantile_cont(z_error, 0.5) AS z_error_p50,
    quantile_cont(z_error, 0.9) AS z_error_p90,
    avg(z_error) AS z_error_mean,
    stddev_samp(z_error) AS z_error_std,

    quantile_cont(yaw_error, 0.5) AS yaw_error_p50,
    quantile_cont(yaw_error, 0.9) AS yaw_error_p90,
    avg(yaw_error) AS yaw_error_mean,
    stddev_samp(yaw_error) AS yaw_error_std

FROM view_eval_flat
WHERE ${topic_label_id_filter:raw} AND source='GT' AND status='TP'
GROUP BY t4dataset_id, topic_name, label, distance_bin
ORDER BY t4dataset_id, topic_name, label, distance_bin;