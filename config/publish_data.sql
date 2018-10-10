insert into trend_suggestion(sku, brand, color, sub_cat_type, trend, updated_at, country, description,   images, modified_trend, product_type)
select sku, brand, color, sub_cat_type, "{0}", '{1}', country,  short_description, image_url, '', product_type
from temp_published_data
where `{0}` = 1
