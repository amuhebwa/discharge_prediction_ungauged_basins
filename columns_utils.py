'''
Notes: 
Order 8:
input_features 263
Remove width related variables: 260

Order 7:
input_features 227
Remove width related variables: 224

Order 6:
input_features 191
Remove width related variables: 188

Order 5:
input_features 155
Remove width related variables: 152

Order 4:
input_features 119
Remove width related variables: 116

'''


# ------------------ ORDER 8 ---------------------------------------------------

order8_better_columns_order = [ 

'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'order5_AirTemp', 'order5_Albedo', 'order5_Avg_Skin_Temp', 'order5_BsGndWtrRunoff', 'order5_CanopyWaterEvpn', 'order5_DirectEvonBareSoil',
'order5_DwdShtWvRadFlux', 'order5_Evapotranspn', 'order5_HeatFlux','order5_LngWaveRadFlux', 'order5_LtHeat', 'order5_NetRadFlux',
'order5_NetShtWvRadFlux', 'order5_PlantCanopyWater','order5_PotEvpnRate', 'order5_Pressure', 'order5_RainPcpRate','order5_RootZoneSoilMstr', 
'order5_Sen.HtFlux', 'order5_SnowDepth', 'order5_SnowDepthWtrEq', 'order5_SnowMelt', 'order5_SnowPcpRate','order5_SoilMst10', 'order5_SoilMst100', 
'order5_SoilMst200', 'order5_SoilMst40', 'order5_SoilTmp10', 'order5_SoilTmp100','order5_SoilTmp200', 'order5_SoilTmp40', 'order5_SpecHmd','order5_StmSurfRunoff', 
'order5_TotalPcpRate', 'order5_Tspn','order5_WindSpd',

'order6_AirTemp', 'order6_Albedo', 'order6_Avg_Skin_Temp', 'order6_BsGndWtrRunoff','order6_CanopyWaterEvpn', 'order6_DirectEvonBareSoil',
'order6_DwdShtWvRadFlux', 'order6_Evapotranspn', 'order6_HeatFlux','order6_LngWaveRadFlux', 'order6_LtHeat', 'order6_NetRadFlux',
'order6_NetShtWvRadFlux', 'order6_PlantCanopyWater','order6_PotEvpnRate', 'order6_Pressure', 'order6_RainPcpRate','order6_RootZoneSoilMstr', 
'order6_Sen.HtFlux', 'order6_SnowDepth','order6_SnowDepthWtrEq', 'order6_SnowMelt', 'order6_SnowPcpRate','order6_SoilMst10', 'order6_SoilMst100', 'order6_SoilMst200',
'order6_SoilMst40', 'order6_SoilTmp10', 'order6_SoilTmp100','order6_SoilTmp200', 'order6_SoilTmp40', 'order6_SpecHmd','order6_StmSurfRunoff', 'order6_TotalPcpRate', 'order6_Tspn',
'order6_WindSpd', 

'order7_AirTemp', 'order7_Albedo', 'order7_Avg_Skin_Temp', 'order7_BsGndWtrRunoff', 'order7_CanopyWaterEvpn', 'order7_DirectEvonBareSoil','order7_DwdShtWvRadFlux', 
'order7_Evapotranspn', 'order7_HeatFlux','order7_LngWaveRadFlux', 'order7_LtHeat', 'order7_NetRadFlux','order7_NetShtWvRadFlux', 'order7_PlantCanopyWater',
'order7_PotEvpnRate', 'order7_Pressure', 'order7_RainPcpRate','order7_RootZoneSoilMstr', 'order7_Sen.HtFlux', 'order7_SnowDepth','order7_SnowDepthWtrEq', 
'order7_SnowMelt', 'order7_SnowPcpRate','order7_SoilMst10', 'order7_SoilMst100', 'order7_SoilMst200','order7_SoilMst40', 'order7_SoilTmp10', 'order7_SoilTmp100',
'order7_SoilTmp200', 'order7_SoilTmp40', 'order7_SpecHmd','order7_StmSurfRunoff', 'order7_TotalPcpRate', 'order7_Tspn','order7_WindSpd', 

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'length','sinuosity', 'slope', 'uparea', 'lengthdir','strmDrop', 
'width_mean', 'max_width', 
'width', 
'discharge',
]

order8_columns_to_scale = [ 
'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'order5_AirTemp', 'order5_Albedo', 'order5_Avg_Skin_Temp', 'order5_BsGndWtrRunoff', 'order5_CanopyWaterEvpn', 'order5_DirectEvonBareSoil',
'order5_DwdShtWvRadFlux', 'order5_Evapotranspn', 'order5_HeatFlux','order5_LngWaveRadFlux', 'order5_LtHeat', 'order5_NetRadFlux',
'order5_NetShtWvRadFlux', 'order5_PlantCanopyWater','order5_PotEvpnRate', 'order5_Pressure', 'order5_RainPcpRate','order5_RootZoneSoilMstr', 
'order5_Sen.HtFlux', 'order5_SnowDepth', 'order5_SnowDepthWtrEq', 'order5_SnowMelt', 'order5_SnowPcpRate','order5_SoilMst10', 'order5_SoilMst100', 
'order5_SoilMst200', 'order5_SoilMst40', 'order5_SoilTmp10', 'order5_SoilTmp100','order5_SoilTmp200', 'order5_SoilTmp40', 'order5_SpecHmd','order5_StmSurfRunoff', 
'order5_TotalPcpRate', 'order5_Tspn','order5_WindSpd',

'order6_AirTemp', 'order6_Albedo', 'order6_Avg_Skin_Temp', 'order6_BsGndWtrRunoff','order6_CanopyWaterEvpn', 'order6_DirectEvonBareSoil',
'order6_DwdShtWvRadFlux', 'order6_Evapotranspn', 'order6_HeatFlux','order6_LngWaveRadFlux', 'order6_LtHeat', 'order6_NetRadFlux',
'order6_NetShtWvRadFlux', 'order6_PlantCanopyWater','order6_PotEvpnRate', 'order6_Pressure', 'order6_RainPcpRate','order6_RootZoneSoilMstr', 
'order6_Sen.HtFlux', 'order6_SnowDepth','order6_SnowDepthWtrEq', 'order6_SnowMelt', 'order6_SnowPcpRate','order6_SoilMst10', 'order6_SoilMst100', 'order6_SoilMst200',
'order6_SoilMst40', 'order6_SoilTmp10', 'order6_SoilTmp100','order6_SoilTmp200', 'order6_SoilTmp40', 'order6_SpecHmd','order6_StmSurfRunoff', 'order6_TotalPcpRate', 'order6_Tspn',
'order6_WindSpd', 

'order7_AirTemp', 'order7_Albedo', 'order7_Avg_Skin_Temp', 'order7_BsGndWtrRunoff', 'order7_CanopyWaterEvpn', 'order7_DirectEvonBareSoil','order7_DwdShtWvRadFlux', 
'order7_Evapotranspn', 'order7_HeatFlux','order7_LngWaveRadFlux', 'order7_LtHeat', 'order7_NetRadFlux','order7_NetShtWvRadFlux', 'order7_PlantCanopyWater',
'order7_PotEvpnRate', 'order7_Pressure', 'order7_RainPcpRate','order7_RootZoneSoilMstr', 'order7_Sen.HtFlux', 'order7_SnowDepth','order7_SnowDepthWtrEq', 
'order7_SnowMelt', 'order7_SnowPcpRate','order7_SoilMst10', 'order7_SoilMst100', 'order7_SoilMst200','order7_SoilMst40', 'order7_SoilTmp10', 'order7_SoilTmp100',
'order7_SoilTmp200', 'order7_SoilTmp40', 'order7_SpecHmd','order7_StmSurfRunoff', 'order7_TotalPcpRate', 'order7_Tspn','order7_WindSpd', 

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'width', 
'discharge',
]

order8_columns_to_decompose = [ 

'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'order5_AirTemp', 'order5_Albedo', 'order5_Avg_Skin_Temp', 'order5_BsGndWtrRunoff', 'order5_CanopyWaterEvpn', 'order5_DirectEvonBareSoil',
'order5_DwdShtWvRadFlux', 'order5_Evapotranspn', 'order5_HeatFlux','order5_LngWaveRadFlux', 'order5_LtHeat', 'order5_NetRadFlux',
'order5_NetShtWvRadFlux', 'order5_PlantCanopyWater','order5_PotEvpnRate', 'order5_Pressure', 'order5_RainPcpRate','order5_RootZoneSoilMstr', 
'order5_Sen.HtFlux', 'order5_SnowDepth', 'order5_SnowDepthWtrEq', 'order5_SnowMelt', 'order5_SnowPcpRate','order5_SoilMst10', 'order5_SoilMst100', 
'order5_SoilMst200', 'order5_SoilMst40', 'order5_SoilTmp10', 'order5_SoilTmp100','order5_SoilTmp200', 'order5_SoilTmp40', 'order5_SpecHmd','order5_StmSurfRunoff', 
'order5_TotalPcpRate', 'order5_Tspn','order5_WindSpd',

'order6_AirTemp', 'order6_Albedo', 'order6_Avg_Skin_Temp', 'order6_BsGndWtrRunoff','order6_CanopyWaterEvpn', 'order6_DirectEvonBareSoil',
'order6_DwdShtWvRadFlux', 'order6_Evapotranspn', 'order6_HeatFlux','order6_LngWaveRadFlux', 'order6_LtHeat', 'order6_NetRadFlux',
'order6_NetShtWvRadFlux', 'order6_PlantCanopyWater','order6_PotEvpnRate', 'order6_Pressure', 'order6_RainPcpRate','order6_RootZoneSoilMstr', 
'order6_Sen.HtFlux', 'order6_SnowDepth','order6_SnowDepthWtrEq', 'order6_SnowMelt', 'order6_SnowPcpRate','order6_SoilMst10', 'order6_SoilMst100', 'order6_SoilMst200',
'order6_SoilMst40', 'order6_SoilTmp10', 'order6_SoilTmp100','order6_SoilTmp200', 'order6_SoilTmp40', 'order6_SpecHmd','order6_StmSurfRunoff', 'order6_TotalPcpRate', 'order6_Tspn',
'order6_WindSpd', 

'order7_AirTemp', 'order7_Albedo', 'order7_Avg_Skin_Temp', 'order7_BsGndWtrRunoff', 'order7_CanopyWaterEvpn', 'order7_DirectEvonBareSoil','order7_DwdShtWvRadFlux', 
'order7_Evapotranspn', 'order7_HeatFlux','order7_LngWaveRadFlux', 'order7_LtHeat', 'order7_NetRadFlux','order7_NetShtWvRadFlux', 'order7_PlantCanopyWater',
'order7_PotEvpnRate', 'order7_Pressure', 'order7_RainPcpRate','order7_RootZoneSoilMstr', 'order7_Sen.HtFlux', 'order7_SnowDepth','order7_SnowDepthWtrEq', 
'order7_SnowMelt', 'order7_SnowPcpRate','order7_SoilMst10', 'order7_SoilMst100', 'order7_SoilMst200','order7_SoilMst40', 'order7_SoilTmp10', 'order7_SoilTmp100',
'order7_SoilTmp200', 'order7_SoilTmp40', 'order7_SpecHmd','order7_StmSurfRunoff', 'order7_TotalPcpRate', 'order7_Tspn','order7_WindSpd', 

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'width', 
]


# -------------------------------- ORDER 7 -----------------------------------------------------------------------------
order7_better_columns_order = [ 

'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'order5_AirTemp', 'order5_Albedo', 'order5_Avg_Skin_Temp', 'order5_BsGndWtrRunoff', 'order5_CanopyWaterEvpn', 'order5_DirectEvonBareSoil',
'order5_DwdShtWvRadFlux', 'order5_Evapotranspn', 'order5_HeatFlux','order5_LngWaveRadFlux', 'order5_LtHeat', 'order5_NetRadFlux',
'order5_NetShtWvRadFlux', 'order5_PlantCanopyWater','order5_PotEvpnRate', 'order5_Pressure', 'order5_RainPcpRate','order5_RootZoneSoilMstr', 
'order5_Sen.HtFlux', 'order5_SnowDepth', 'order5_SnowDepthWtrEq', 'order5_SnowMelt', 'order5_SnowPcpRate','order5_SoilMst10', 'order5_SoilMst100', 
'order5_SoilMst200', 'order5_SoilMst40', 'order5_SoilTmp10', 'order5_SoilTmp100','order5_SoilTmp200', 'order5_SoilTmp40', 'order5_SpecHmd','order5_StmSurfRunoff', 
'order5_TotalPcpRate', 'order5_Tspn','order5_WindSpd',

'order6_AirTemp', 'order6_Albedo', 'order6_Avg_Skin_Temp', 'order6_BsGndWtrRunoff','order6_CanopyWaterEvpn', 'order6_DirectEvonBareSoil',
'order6_DwdShtWvRadFlux', 'order6_Evapotranspn', 'order6_HeatFlux','order6_LngWaveRadFlux', 'order6_LtHeat', 'order6_NetRadFlux',
'order6_NetShtWvRadFlux', 'order6_PlantCanopyWater','order6_PotEvpnRate', 'order6_Pressure', 'order6_RainPcpRate','order6_RootZoneSoilMstr', 
'order6_Sen.HtFlux', 'order6_SnowDepth','order6_SnowDepthWtrEq', 'order6_SnowMelt', 'order6_SnowPcpRate','order6_SoilMst10', 'order6_SoilMst100', 'order6_SoilMst200',
'order6_SoilMst40', 'order6_SoilTmp10', 'order6_SoilTmp100','order6_SoilTmp200', 'order6_SoilTmp40', 'order6_SpecHmd','order6_StmSurfRunoff', 'order6_TotalPcpRate', 'order6_Tspn',
'order6_WindSpd', 

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'length','sinuosity', 'slope', 'uparea', 'lengthdir','strmDrop', 
'width_mean', 'max_width', 
'width', 
'discharge',
]

order7_columns_to_scale = [ 
'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'order5_AirTemp', 'order5_Albedo', 'order5_Avg_Skin_Temp', 'order5_BsGndWtrRunoff', 'order5_CanopyWaterEvpn', 'order5_DirectEvonBareSoil',
'order5_DwdShtWvRadFlux', 'order5_Evapotranspn', 'order5_HeatFlux','order5_LngWaveRadFlux', 'order5_LtHeat', 'order5_NetRadFlux',
'order5_NetShtWvRadFlux', 'order5_PlantCanopyWater','order5_PotEvpnRate', 'order5_Pressure', 'order5_RainPcpRate','order5_RootZoneSoilMstr', 
'order5_Sen.HtFlux', 'order5_SnowDepth', 'order5_SnowDepthWtrEq', 'order5_SnowMelt', 'order5_SnowPcpRate','order5_SoilMst10', 'order5_SoilMst100', 
'order5_SoilMst200', 'order5_SoilMst40', 'order5_SoilTmp10', 'order5_SoilTmp100','order5_SoilTmp200', 'order5_SoilTmp40', 'order5_SpecHmd','order5_StmSurfRunoff', 
'order5_TotalPcpRate', 'order5_Tspn','order5_WindSpd',

'order6_AirTemp', 'order6_Albedo', 'order6_Avg_Skin_Temp', 'order6_BsGndWtrRunoff','order6_CanopyWaterEvpn', 'order6_DirectEvonBareSoil',
'order6_DwdShtWvRadFlux', 'order6_Evapotranspn', 'order6_HeatFlux','order6_LngWaveRadFlux', 'order6_LtHeat', 'order6_NetRadFlux',
'order6_NetShtWvRadFlux', 'order6_PlantCanopyWater','order6_PotEvpnRate', 'order6_Pressure', 'order6_RainPcpRate','order6_RootZoneSoilMstr', 
'order6_Sen.HtFlux', 'order6_SnowDepth','order6_SnowDepthWtrEq', 'order6_SnowMelt', 'order6_SnowPcpRate','order6_SoilMst10', 'order6_SoilMst100', 'order6_SoilMst200',
'order6_SoilMst40', 'order6_SoilTmp10', 'order6_SoilTmp100','order6_SoilTmp200', 'order6_SoilTmp40', 'order6_SpecHmd','order6_StmSurfRunoff', 'order6_TotalPcpRate', 'order6_Tspn',
'order6_WindSpd', 

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'width', 
'discharge',
]

order7_columns_to_decompose = [ 

'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'order5_AirTemp', 'order5_Albedo', 'order5_Avg_Skin_Temp', 'order5_BsGndWtrRunoff', 'order5_CanopyWaterEvpn', 'order5_DirectEvonBareSoil',
'order5_DwdShtWvRadFlux', 'order5_Evapotranspn', 'order5_HeatFlux','order5_LngWaveRadFlux', 'order5_LtHeat', 'order5_NetRadFlux',
'order5_NetShtWvRadFlux', 'order5_PlantCanopyWater','order5_PotEvpnRate', 'order5_Pressure', 'order5_RainPcpRate','order5_RootZoneSoilMstr', 
'order5_Sen.HtFlux', 'order5_SnowDepth', 'order5_SnowDepthWtrEq', 'order5_SnowMelt', 'order5_SnowPcpRate','order5_SoilMst10', 'order5_SoilMst100', 
'order5_SoilMst200', 'order5_SoilMst40', 'order5_SoilTmp10', 'order5_SoilTmp100','order5_SoilTmp200', 'order5_SoilTmp40', 'order5_SpecHmd','order5_StmSurfRunoff', 
'order5_TotalPcpRate', 'order5_Tspn','order5_WindSpd',

'order6_AirTemp', 'order6_Albedo', 'order6_Avg_Skin_Temp', 'order6_BsGndWtrRunoff','order6_CanopyWaterEvpn', 'order6_DirectEvonBareSoil',
'order6_DwdShtWvRadFlux', 'order6_Evapotranspn', 'order6_HeatFlux','order6_LngWaveRadFlux', 'order6_LtHeat', 'order6_NetRadFlux',
'order6_NetShtWvRadFlux', 'order6_PlantCanopyWater','order6_PotEvpnRate', 'order6_Pressure', 'order6_RainPcpRate','order6_RootZoneSoilMstr', 
'order6_Sen.HtFlux', 'order6_SnowDepth','order6_SnowDepthWtrEq', 'order6_SnowMelt', 'order6_SnowPcpRate','order6_SoilMst10', 'order6_SoilMst100', 'order6_SoilMst200',
'order6_SoilMst40', 'order6_SoilTmp10', 'order6_SoilTmp100','order6_SoilTmp200', 'order6_SoilTmp40', 'order6_SpecHmd','order6_StmSurfRunoff', 'order6_TotalPcpRate', 'order6_Tspn',
'order6_WindSpd', 

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'width', 
]

# -------------------------------- ORDER 6 -----------------------------------------------------------------------------
order6_better_columns_order = [ 

'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'order5_AirTemp', 'order5_Albedo', 'order5_Avg_Skin_Temp', 'order5_BsGndWtrRunoff', 'order5_CanopyWaterEvpn', 'order5_DirectEvonBareSoil',
'order5_DwdShtWvRadFlux', 'order5_Evapotranspn', 'order5_HeatFlux','order5_LngWaveRadFlux', 'order5_LtHeat', 'order5_NetRadFlux',
'order5_NetShtWvRadFlux', 'order5_PlantCanopyWater','order5_PotEvpnRate', 'order5_Pressure', 'order5_RainPcpRate','order5_RootZoneSoilMstr', 
'order5_Sen.HtFlux', 'order5_SnowDepth', 'order5_SnowDepthWtrEq', 'order5_SnowMelt', 'order5_SnowPcpRate','order5_SoilMst10', 'order5_SoilMst100', 
'order5_SoilMst200', 'order5_SoilMst40', 'order5_SoilTmp10', 'order5_SoilTmp100','order5_SoilTmp200', 'order5_SoilTmp40', 'order5_SpecHmd','order5_StmSurfRunoff', 
'order5_TotalPcpRate', 'order5_Tspn','order5_WindSpd',

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'length','sinuosity', 'slope', 'uparea', 'lengthdir','strmDrop', 
'width_mean', 'max_width', 
'width', 
'discharge',
]

order6_columns_to_scale = [ 
'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'order5_AirTemp', 'order5_Albedo', 'order5_Avg_Skin_Temp', 'order5_BsGndWtrRunoff', 'order5_CanopyWaterEvpn', 'order5_DirectEvonBareSoil',
'order5_DwdShtWvRadFlux', 'order5_Evapotranspn', 'order5_HeatFlux','order5_LngWaveRadFlux', 'order5_LtHeat', 'order5_NetRadFlux',
'order5_NetShtWvRadFlux', 'order5_PlantCanopyWater','order5_PotEvpnRate', 'order5_Pressure', 'order5_RainPcpRate','order5_RootZoneSoilMstr', 
'order5_Sen.HtFlux', 'order5_SnowDepth', 'order5_SnowDepthWtrEq', 'order5_SnowMelt', 'order5_SnowPcpRate','order5_SoilMst10', 'order5_SoilMst100', 
'order5_SoilMst200', 'order5_SoilMst40', 'order5_SoilTmp10', 'order5_SoilTmp100','order5_SoilTmp200', 'order5_SoilTmp40', 'order5_SpecHmd','order5_StmSurfRunoff', 
'order5_TotalPcpRate', 'order5_Tspn','order5_WindSpd',


'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'width', 
'discharge',
]

order6_columns_to_decompose = [ 

'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'order5_AirTemp', 'order5_Albedo', 'order5_Avg_Skin_Temp', 'order5_BsGndWtrRunoff', 'order5_CanopyWaterEvpn', 'order5_DirectEvonBareSoil',
'order5_DwdShtWvRadFlux', 'order5_Evapotranspn', 'order5_HeatFlux','order5_LngWaveRadFlux', 'order5_LtHeat', 'order5_NetRadFlux',
'order5_NetShtWvRadFlux', 'order5_PlantCanopyWater','order5_PotEvpnRate', 'order5_Pressure', 'order5_RainPcpRate','order5_RootZoneSoilMstr', 
'order5_Sen.HtFlux', 'order5_SnowDepth', 'order5_SnowDepthWtrEq', 'order5_SnowMelt', 'order5_SnowPcpRate','order5_SoilMst10', 'order5_SoilMst100', 
'order5_SoilMst200', 'order5_SoilMst40', 'order5_SoilTmp10', 'order5_SoilTmp100','order5_SoilTmp200', 'order5_SoilTmp40', 'order5_SpecHmd','order5_StmSurfRunoff', 
'order5_TotalPcpRate', 'order5_Tspn','order5_WindSpd',

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'width', 
]

# -------------------------------- ORDER 5 -----------------------------------------------------------------------------
order5_better_columns_order = [ 

'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'length','sinuosity', 'slope', 'uparea', 'lengthdir','strmDrop', 
'width_mean', 'max_width', 
'width', 
'discharge',
]

order5_columns_to_scale = [ 
'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'width', 
'discharge',
]

order5_columns_to_decompose = [ 

'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'order4_AirTemp', 'order4_Albedo', 'order4_Avg_Skin_Temp', 'order4_BsGndWtrRunoff','order4_CanopyWaterEvpn', 'order4_DirectEvonBareSoil',
'order4_DwdShtWvRadFlux', 'order4_Evapotranspn', 'order4_HeatFlux','order4_LngWaveRadFlux', 'order4_LtHeat', 'order4_NetRadFlux',
'order4_NetShtWvRadFlux', 'order4_PlantCanopyWater','order4_PotEvpnRate', 'order4_Pressure', 'order4_RainPcpRate','order4_RootZoneSoilMstr',
'order4_Sen.HtFlux', 'order4_SnowDepth', 'order4_SnowDepthWtrEq', 'order4_SnowMelt', 'order4_SnowPcpRate','order4_SoilMst10',
'order4_SoilMst100', 'order4_SoilMst200','order4_SoilMst40', 'order4_SoilTmp10', 'order4_SoilTmp100','order4_SoilTmp200','order4_SoilTmp40',
'order4_SpecHmd', 'order4_StmSurfRunoff', 'order4_TotalPcpRate', 'order4_Tspn', 'order4_WindSpd',

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'width', 
]

# -------------------------------- ORDER 4 -----------------------------------------------------------------------------
order4_better_columns_order = [ 
'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'length','sinuosity', 'slope', 'uparea', 'lengthdir','strmDrop', 
'width_mean', 'max_width', 
'width', 
'discharge',
]

order4_columns_to_scale = [ 
'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'width', 
'discharge',
]

order4_columns_to_decompose = [ 
'order3_AirTemp', 'order3_Albedo', 'order3_Avg_Skin_Temp', 'order3_BsGndWtrRunoff', 'order3_CanopyWaterEvpn', 'order3_DirectEvonBareSoil',
'order3_DwdShtWvRadFlux', 'order3_Evapotranspn', 'order3_HeatFlux','order3_LngWaveRadFlux', 'order3_LtHeat', 'order3_NetRadFlux',
'order3_NetShtWvRadFlux', 'order3_PlantCanopyWater','order3_PotEvpnRate', 'order3_Pressure', 'order3_RainPcpRate',
'order3_RootZoneSoilMstr', 'order3_Sen.HtFlux', 'order3_SnowDepth', 'order3_SnowDepthWtrEq', 'order3_SnowMelt', 'order3_SnowPcpRate',
'order3_SoilMst10', 'order3_SoilMst100', 'order3_SoilMst200', 'order3_SoilMst40', 'order3_SoilTmp10', 'order3_SoilTmp100',
'order3_SoilTmp200', 'order3_SoilTmp40', 'order3_SpecHmd', 'order3_StmSurfRunoff', 'order3_TotalPcpRate', 'order3_Tspn', 'order3_WindSpd', 

'basin_Albedo', 'basin_Avg_Skin_Temp','basin_PlantCanopyWater', 'basin_CanopyWaterEvpn','basin_DirectEvonBareSoil', 'basin_Evapotranspn',
'basin_LngWaveRadFlux', 'basin_NetRadFlux', 'basin_PotEvpnRate','basin_Pressure', 'basin_SpecHmd', 'basin_HeatFlux','basin_Sen.HtFlux', 
'basin_LtHeat', 'basin_StmSurfRunoff','basin_BsGndWtrRunoff', 'basin_SnowMelt', 'basin_TotalPcpRate','basin_RainPcpRate', 'basin_RootZoneSoilMstr',
'basin_SnowDepthWtrEq', 'basin_DwdShtWvRadFlux', 'basin_SnowDepth','basin_SnowPcpRate', 'basin_SoilMst10', 'basin_SoilMst40',
'basin_SoilMst100', 'basin_SoilMst200', 'basin_SoilTmp10','basin_SoilTmp40', 'basin_SoilTmp100', 'basin_SoilTmp200','basin_NetShtWvRadFlux',
'basin_AirTemp', 'basin_Tspn', 'basin_WindSpd',

'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure','SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat', 'StmSurfRunoff',
'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate','RootZoneSoilMstr', 'SnowDepthWtrEq', 'DwdShtWvRadFlux',
'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200',
'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI', 

'width', 
]


order4_dict = {
       'better_columns_order':order4_better_columns_order,
       'columns_to_scale':order4_columns_to_scale,
       'columns_to_decompose':order4_columns_to_decompose
}


order5_dict = {
       'better_columns_order':order5_better_columns_order,
       'columns_to_scale':order5_columns_to_scale,
       'columns_to_decompose':order5_columns_to_decompose
}
order6_dict = {
       'better_columns_order':order6_better_columns_order,
       'columns_to_scale':order6_columns_to_scale,
       'columns_to_decompose':order6_columns_to_decompose
}

order7_dict = {
       'better_columns_order':order7_better_columns_order,
       'columns_to_scale':order7_columns_to_scale,
       'columns_to_decompose':order7_columns_to_decompose
}

order8_dict = {
       'better_columns_order':order8_better_columns_order,
       'columns_to_scale':order8_columns_to_scale,
       'columns_to_decompose':order8_columns_to_decompose
}

orders_lookuptable = {
        '4':order4_dict,
        '5':order5_dict,
        '6':order6_dict,
        '7':order7_dict,
        '8':order8_dict
}

