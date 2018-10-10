WITH dim_product AS (
        SELECT DISTINCT sku_config as sku
        FROM edw.dim_product
        where valid_to_date > '9999-01-01'
        GROUP BY sku_config
)
SELECT cc.sku,
        /* cc.sku_supplier_config,    */  
        -- cc.status,
        -- cc.is_recent, 
        cc.brand,
        meta.country,
        -- cc.brand_department,
        cc.department,
        cc.product_name,
        -- cc.buying_cat_type,
        cc.buying_planning_cat_type,
        -- cc.crm_category,
        cc.category,
        cc.sub_cat_type,
        cc.occasion,
        cc.color,
        cc.color_family,
        cc.short_description,  
        cc.catalog_attribute_set_name,
        cc.catalog_attribute_set_label,
        -- cc.catalog_type,
        -- cc.season,
        -- cc.season_year,
        -- cc.created_at,
        -- cc.first_visible_at,
        cc.activated_at,
        meta.days_since_activation_date,    
        -- meta.last_visible_at,
        -- meta.size_availability,
        cc.image_url,
        -- cc.lq_image_url,        
        -- cc.product_url,
        -- cc.is_pack,
        -- cc.is_returnable,
        -- cc.nonsale_item,
        -- cc.shop_type,
        -- cc.venture_code, 
        -- cc.description,
        -- cc.brand_status,
        'retail' as product_type
FROM edw.catalog_config cc
INNER JOIN datascience.ape_metadata meta
        ON cc.sku = meta.sku
        AND cc.venture_code = 'sg'
        AND meta.country = 'sg'
        AND cc.gender = 'Female'       
        AND cc.status NOT IN ('inactive', 'deleted')      
        AND cc.brand_status = 'active'        
        AND meta.size_availability > '0'       
        AND cc.buying_planning_cat_type = 'Wapp' 
INNER JOIN dim_product dp
        ON cc.sku = dp.sku
-- WHERE meta.days_since_activation_date <= 180
		
        
/* 
SELECT cc.sku, 
        cc.sku_supplier_config,        
        cc.brand, 
        cc.product_name,
        cc.status,
        cc.buying_planning_cat_type,
        cc.crm_category,
        cc.category,
        cc.sub_cat_type,         
        cc.short_description,        
        cc.color,
        cc.color_family,         
        cc.description,
        cc.image_url,
        cc.lq_image_url,        
        cc.product_url,
        cc.brand_department,
        meta.size_availability, 
        meta.days_since_activation_date,    
        meta.last_visible_at       
FROM edw.catalog_config cc
INNER JOIN datascience.ape_metadata meta
        ON cc.sku = meta.sku
        AND cc.venture_code = 'sg'
        AND meta.country = 'sg'
        AND cc.gender = 'Female'        
        AND cc.status NOT IN ('inactive', 'deleted')       
        AND cc.brand_status = 'active'        
        AND meta.size_availability > '0'       
        AND cc.buying_planning_cat_type NOT IN ('Beau')
*/