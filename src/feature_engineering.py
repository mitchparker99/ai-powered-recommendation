import numpy as np


def create_features(df):
    df['user_product_interaction'] = df.groupby(
        'Clothing ID')['Rating'].transform('mean')
    return df
