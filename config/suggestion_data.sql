SELECT DISTINCT modified_trend AS name, ts.sku 
FROM trend_suggestion ts 
        INNER JOIN (
              SELECT sku, MAX(updated_at) AS updated_at
              FROM trend_suggestion
              GROUP BY sku
        ) latest ON ts.sku = latest.sku AND ts.updated_at = latest.updated_at
WHERE modified_trend IS NOT NULL 
/* TREND_TAGGING_FILTER */
        AND ts.sku NOT IN (
                SELECT DISTINCT sku
                FROM trend_tagging.trend_tagging_raw
                ) 