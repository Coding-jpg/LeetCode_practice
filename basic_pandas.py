'''
595
'''
import pandas as pd

def big_countries(world: pd.DataFrame) -> pd.DataFrame:
    print(world.head())
    return world[(world['area'] >= 3000000) | (world['population'] >= 25000000)][['name','population','area']]

'''
1757
'''
import pandas as pd

def find_products(products: pd.DataFrame) -> pd.DataFrame:
    return products.loc[(products['low_fats'] == 'Y') & (products['recyclable'] == 'Y'), ['product_id']]
    