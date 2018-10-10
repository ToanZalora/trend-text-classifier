SELECT DISTINCT category_name AS name, ts.sku 
FROM trend_tagging_raw ts 
        INNER JOIN (
              SELECT sku, MAX(updated_at) AS updated_at
              FROM trend_tagging_raw
              GROUP BY sku
        ) latest ON ts.sku = latest.sku AND ts.updated_at = latest.updated_at