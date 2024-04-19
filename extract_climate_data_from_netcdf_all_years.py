import argparse
import os
import pandas as pd
import numpy as np
import glob
from netCDF4 import Dataset, num2date
from datetime import datetime as dt
import sys 
import code


# Get the index of longitude and latitude
# Ref: https://stackoverflow.com/questions/33789379/netcdf-and-python-finding-the-closest-lon-lat-index-given-actual-lon-lat-values
def get_index(_val, _array):
    return (np.abs(_array - _val)).argmin()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='File Parameters')
    parser.add_argument('--chunk_id', required=True)
    args = parser.parse_args()
    dir_name = '/gypsum/eguide/projects/amuhebwa/RiversPrediction'
    chunk_index = args.chunk_id
    chunk_index = int(chunk_index)

    basins_polygons_df = pd.read_csv(f'{dir_name}/reaches_basins_polygons.csv')
    polygon_df_chunks = np.array_split(basins_polygons_df, 100)
    reaches_subset_df = polygon_df_chunks[chunk_index]
    parent_dir = '{}/glads_files'.format(dir_name)
    parentfolders = [ f.path for f in os.scandir(parent_dir) if f.is_dir() ]

    allYears_merged_list = list()
    for _, year_folder in enumerate(parentfolders):
        current_files = sorted(glob.glob('{}/*.nc4'.format(year_folder)))

        lats = reaches_subset_df.lat.values
        lons = reaches_subset_df.lon.values
        reaches_comids = reaches_subset_df.COMID.values

        #cnt = 0
        threeHourlyDfs_list = []
        for _, current_ncdf in enumerate(current_files):
            nc_data = Dataset(current_ncdf, 'r')
            arr_Albedo_inst = []
            arr_AvgSurfT_inst= []
            arr_CanopInt_inst = []
            arr_ECanop_tavg = []
            arr_ESoil_tavg = []
            arr_Evap_tavg = []
            arr_LWdown_f_tavg = []
            arr_Lwnet_tavg = []
            arr_PotEvap_tavg = []
            arr_Psurf_f_inst = []
            arr_Qair_f_inst = []
            arr_Qg_tavg = []
            arr_Qh_tavg = []
            arr_Qle_tavg = []
            arr_Qs_acc = []
            arr_Qsb_acc = []
            arr_Qsm_acc = []
            arr_Rainf_f_tavg = []
            arr_Rainf_tavg = []
            arr_RootMoist_inst = []
            arr_SWE_inst = []
            arr_SWdown_f_tavg = []
            arr_SnowDepth_inst = []
            arr_Snowf_tavg = []
            arr_SoilMoi0_10cm_inst = []
            arr_SoilMoi10_40cm_inst = []
            arr_SoilMoi40_100cm_inst = []
            arr_SoilMoi100_200cm_inst = []
            arr_SoilTMP0_10cm_inst = []
            arr_SoilTMP10_40cm_inst = []
            arr_SoilTMP40_100cm_inst = []
            arr_SoilTMP100_200cm_inst = []
            arr_Swnet_tavg = []
            arr_Tair_f_inst = []
            arr_Tveg_tavg = []
            arr_Wind_f_inst = []
            arr_year = []
            arr_month = []
            arr_day = []
            arr_comids = []
            #cnt += 1 

            for lat, lon, reach_comid in zip(lats, lons, reaches_comids):
                times = num2date(nc_data.variables['time'][:], nc_data.variables['time'].units, nc_data.variables['time'].calendar)[:]
                times_formatted = [dt(x.year,x.month,x.day,x.hour,x.minute) for x in times]
                day_of_month = times_formatted[0].day
                current_year = times_formatted[0].year
                current_month = times_formatted[0].month

                nc_lats = nc_data['lat'][:]
                nc_lons = nc_data['lon'][:]
                lat_index = get_index(lat, nc_lats)
                lon_index = get_index(lon, nc_lons)

                arr_comids.append(reach_comid)
                arr_month.append(current_month)
                arr_year.append(current_year)
                arr_day.append(day_of_month)
                arr_Albedo_inst.append(nc_data['Albedo_inst'][:, lat_index, lon_index].data[0])
                arr_AvgSurfT_inst.append(nc_data['AvgSurfT_inst'][:, lat_index, lon_index].data[0])
                arr_CanopInt_inst.append(nc_data['CanopInt_inst'][:, lat_index, lon_index].data[0])
                arr_ECanop_tavg.append(nc_data['ECanop_tavg'][:, lat_index, lon_index].data[0])
                arr_ESoil_tavg.append(nc_data['ESoil_tavg'][:, lat_index, lon_index].data[0])
                arr_Evap_tavg.append(nc_data['Evap_tavg'][:, lat_index, lon_index].data[0])
                arr_LWdown_f_tavg.append(nc_data['LWdown_f_tavg'][:, lat_index, lon_index].data[0])
                arr_Lwnet_tavg.append(nc_data['Lwnet_tavg'][:, lat_index, lon_index].data[0])
                arr_PotEvap_tavg.append(nc_data['PotEvap_tavg'][:, lat_index, lon_index].data[0])
                arr_Psurf_f_inst.append(nc_data['Psurf_f_inst'][:, lat_index, lon_index].data[0])
                arr_Qair_f_inst.append(nc_data['Qair_f_inst'][:, lat_index, lon_index].data[0])
                arr_Qg_tavg.append(nc_data['Qg_tavg'][:, lat_index, lon_index].data[0])
                arr_Qh_tavg.append(nc_data['Qh_tavg'][:, lat_index, lon_index].data[0])
                arr_Qle_tavg.append(nc_data['Qle_tavg'][:, lat_index, lon_index].data[0])
                arr_Qs_acc.append(nc_data['Qs_acc'][:, lat_index, lon_index].data[0])
                arr_Qsb_acc.append(nc_data['Qsb_acc'][:, lat_index, lon_index].data[0])
                arr_Qsm_acc.append(nc_data['Qsm_acc'][:, lat_index, lon_index].data[0])
                arr_Rainf_f_tavg.append(nc_data['Rainf_f_tavg'][:, lat_index, lon_index].data[0])
                arr_Rainf_tavg.append(nc_data['Rainf_tavg'][:, lat_index, lon_index].data[0])
                arr_RootMoist_inst.append(nc_data['RootMoist_inst'][:, lat_index, lon_index].data[0])
                arr_SWE_inst.append(nc_data['SWE_inst'][:, lat_index, lon_index].data[0])
                arr_SWdown_f_tavg.append(nc_data['SWdown_f_tavg'][:, lat_index, lon_index].data[0])
                arr_SnowDepth_inst.append(nc_data['SnowDepth_inst'][:, lat_index, lon_index].data[0])
                arr_Snowf_tavg.append(nc_data['Snowf_tavg'][:, lat_index, lon_index].data[0])
                arr_SoilMoi0_10cm_inst.append(nc_data['SoilMoi0_10cm_inst'][:, lat_index, lon_index].data[0])
                arr_SoilMoi10_40cm_inst.append(nc_data['SoilMoi10_40cm_inst'][:, lat_index, lon_index].data[0])
                arr_SoilMoi40_100cm_inst.append(nc_data['SoilMoi40_100cm_inst'][:, lat_index, lon_index].data[0])
                arr_SoilMoi100_200cm_inst.append(nc_data['SoilMoi100_200cm_inst'][:, lat_index, lon_index].data[0])
                arr_SoilTMP0_10cm_inst.append(nc_data['SoilTMP0_10cm_inst'][:, lat_index, lon_index].data[0])
                arr_SoilTMP10_40cm_inst.append(nc_data['SoilTMP10_40cm_inst'][:, lat_index, lon_index].data[0])
                arr_SoilTMP40_100cm_inst.append(nc_data['SoilTMP40_100cm_inst'][:, lat_index, lon_index].data[0])
                arr_SoilTMP100_200cm_inst.append(nc_data['SoilTMP100_200cm_inst'][:, lat_index, lon_index].data[0])
                arr_Swnet_tavg.append(nc_data['Swnet_tavg'][:, lat_index, lon_index].data[0])
                arr_Tair_f_inst.append(nc_data['Tair_f_inst'][:, lat_index, lon_index].data[0])
                arr_Tveg_tavg.append(nc_data['Tveg_tavg'][:, lat_index, lon_index].data[0])
                arr_Wind_f_inst.append(nc_data['Wind_f_inst'][:, lat_index, lon_index].data[0])

            temp_df = pd.DataFrame()
            temp_df['COMID'] = arr_comids
            temp_df['Year'] = arr_year
            temp_df['Month'] =arr_month
            temp_df['Day'] = arr_day
            temp_df['Albedo_inst'] = arr_Albedo_inst
            temp_df['AvgSurfT_inst'] = arr_AvgSurfT_inst
            temp_df['CanopInt_inst'] = arr_CanopInt_inst
            temp_df['ECanop_tavg'] = arr_ECanop_tavg
            temp_df['ESoil_tavg'] = arr_ESoil_tavg
            temp_df['Evap_tavg'] = arr_Evap_tavg
            temp_df['LWdown_f_tavg'] = arr_LWdown_f_tavg
            temp_df['Lwnet_tavg'] = arr_Lwnet_tavg
            temp_df['PotEvap_tavg'] = arr_PotEvap_tavg
            temp_df['Psurf_f_inst'] = arr_Psurf_f_inst
            temp_df['Qair_f_inst'] = arr_Qair_f_inst
            temp_df['Qg_tavg'] = arr_Qg_tavg
            temp_df['Qh_tavg'] = arr_Qh_tavg
            temp_df['Qle_tavg'] = arr_Qle_tavg
            temp_df['Qs_acc'] = arr_Qs_acc
            temp_df['Qsb_acc'] = arr_Qsb_acc
            temp_df['Qsm_acc'] = arr_Qsm_acc
            temp_df['Rainf_f_tavg'] = arr_Rainf_f_tavg
            temp_df['Rainf_tavg'] = arr_Rainf_tavg
            temp_df['RootMoist_inst'] = arr_RootMoist_inst
            temp_df['SWE_inst'] = arr_SWE_inst
            temp_df['SWdown_f_tavg'] = arr_SWdown_f_tavg
            temp_df['SnowDepth_inst'] = arr_SnowDepth_inst
            temp_df['Snowf_tavg'] = arr_Snowf_tavg
            temp_df['SoilMoi0_10cm_inst'] = arr_SoilMoi0_10cm_inst
            temp_df['SoilMoi10_40cm_inst'] = arr_SoilMoi10_40cm_inst
            temp_df['SoilMoi40_100cm_inst'] = arr_SoilMoi40_100cm_inst
            temp_df['SoilMoi100_200cm_inst'] = arr_SoilMoi100_200cm_inst
            temp_df['SoilTMP0_10cm_inst'] = arr_SoilTMP0_10cm_inst
            temp_df['SoilTMP10_40cm_inst'] = arr_SoilTMP10_40cm_inst
            temp_df['SoilTMP40_100cm_inst'] = arr_SoilTMP40_100cm_inst
            temp_df['SoilTMP100_200cm_inst'] = arr_SoilTMP100_200cm_inst
            temp_df['Swnet_tavg'] = arr_Swnet_tavg
            temp_df['Tair_f_inst'] = arr_Tair_f_inst
            temp_df['Tveg_tavg'] = arr_Tveg_tavg
            temp_df['Wind_f_inst'] = arr_Wind_f_inst
            threeHourlyDfs_list.append(temp_df)

            #if cnt >=24:
            #    break

        if len(threeHourlyDfs_list) > 0:
            oneYear_merged_df = pd.concat(threeHourlyDfs_list, axis=0)
            oneYear_merged_df['Date'] = pd.to_datetime(oneYear_merged_df[['Year', 'Month', 'Day']])
            oneYear_merged_df = oneYear_merged_df.drop(['Year', 'Month', 'Day'], axis=1)
            oneYear_merged_df = oneYear_merged_df.sort_values(by=["Date"])
            #oneYear_merged_df = oneYear_merged_df.groupby(['Date']).mean()
            #oneYear_merged_df = oneYear_merged_df.reset_index()
            name2save=f'{dir_name}/all_climate_data/year_{current_year}_chunkId{chunk_index}.csv'
            oneYear_merged_df.to_csv(name2save, index=False)
            print('Saved Chunk {} for year {}'.format(chunk_index, current_year))




