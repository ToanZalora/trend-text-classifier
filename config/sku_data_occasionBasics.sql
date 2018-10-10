WITH dim_product AS (
        SELECT DISTINCT sku_config as sku
        FROM edw.dim_product
        WHERE valid_to_date > '9999-01-01'
        GROUP BY sku_config
)
SELECT cc.sku,
        cc.brand,
        cc.color,
        cc.sub_cat_type,
        meta.country,
        cc.activated_at,
        meta.days_since_activation_date,    
        cc.image_url,
        cc.short_description,  
        cc.occasion as summary
FROM edw.catalog_config cc
INNER JOIN datascience.ape_metadata meta
        ON cc.sku = meta.sku
        AND cc.venture_code = 'sg'
        AND meta.country = 'sg'     
        AND cc.gender NOT IN ('Male')   
        AND cc.status NOT IN ('inactive', 'deleted')      
        AND cc.brand_status = 'active'        
        AND meta.size_availability > '0'  
        AND cc.occasion LIKE '%Basics%'
		AND cc.buying_planning_cat_type = 'Wapp' 
INNER JOIN dim_product dp
        ON cc.sku = dp.sku
WHERE meta.days_since_activation_date <= 180
		
 