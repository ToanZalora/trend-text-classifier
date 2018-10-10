import pandas as pd

def resolve_color(r, 
				  external_color: pd.DataFrame) -> str:
    if external_color:
        try:
            e_row  =  external_color.loc[r.sku]
            if e_row.color_prob > 0.7:
                return e_row.color
        except:
            pass
    return str(r.color).lower()