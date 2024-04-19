"""
Based on Chaopeng Shen's paper, we create dimensionless discharge
However, we will use precipitation, slope and upstream area.
https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2019WR026793


To normalize discharge, considering the effect of Total precipitation rate, rain precipitation rate, temperature, and Snowmelt, you should normalize discharge with snowmelt and temperature.

Snowmelt is the primary source of water for rivers in cold regions like the Mackenzie basin. Temperature controls the timing and rate of snowmelt. Therefore, normalizing discharge with snowmelt and temperature will help to account for the variability in discharge that is caused by these two factors.

The other two factors, total precipitation rate and rain precipitation rate, are less important for normalizing discharge in cold regions. Total precipitation rate includes both rain and snow, but snowmelt is the primary source of water for rivers in cold regions. Rain precipitation rate is even less important because most of the precipitation in cold regions falls as snow.

Air temperature typically has a broader and more consistent impact on river discharge, especially in regions where snowmelt, glacial melt, or precipitation patterns play a significant role in determining river flow.
"""
import code

import pandas as pd
import numpy as np
import glob


def compute_normalized_discharge(row, epsilon=1e-6):
    """
    Compute normalized discharge for a given row.
    """
    # denominator = row['TotalPcpRate'] * row['slope'] * row['uparea'] + epsilon
    denominator = row['TotalPcpRate'] * row['slope'] * row['uparea'] + epsilon
    return row['discharge'] / denominator


def transform_cols(v):
    return np.log10(np.sqrt(v) + 0.1)


if __name__ == "__main__":
    data_dir = "/gypsum/eguide/projects/amuhebwa/rivers_ML/distributed_complete_dataset/"
    data_files = glob.glob(f'{data_dir}/*.csv')
    epsilon = 1e-6

    for idx, datafile in enumerate(data_files):
        print(f'Processing {datafile}')



        df = pd.read_csv(datafile)

        # Filter out NaN discharge values
        df = df[~df['discharge'].isna()]
        original_discharge = df['discharge'].values
        mean_upstream_area = df['uparea'].mean()
        mean_slope = df['slope'].mean()
        mean_snowmelt = df['SnowMelt'].mean()
        mean_temp = df['AirTemp'].mean()
        # normalize discharge with snowmelt and temperature
        df['discharge'] = df['discharge'] / (mean_snowmelt * mean_temp * mean_slope * mean_upstream_area + epsilon)
        df['discharge'] = df['discharge'].apply(transform_cols)
        df['actual_discharge'] = original_discharge

        # Do the same for Q
        original_Q = df['Q'].values
        df['Q'] = df['Q'] / (mean_snowmelt * mean_temp * mean_upstream_area + epsilon)
        df['Q'] = df['Q'].apply(transform_cols)
        df['actual_Q'] = original_Q
        #save the file
        df.to_csv(datafile, index=False)
        print(f'Finished processing {idx} : {datafile}')
    print('Done!')
