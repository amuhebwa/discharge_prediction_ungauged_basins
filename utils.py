'''
common utils across all files.
An example of common utils include the generated combinatins of sets

'''
all_stations_comids = [
    82026688, 82014643, 82036808, 82039908, 82005809, 82022245, 82031602, 82042445, 82029872, 82011128,
    82018092, 82039999, 82018084, 82041599, 82028307, 82025133, 82035594, 82041573, 82039917, 82025057,
    82029922, 82008607, 82022185, 82028537, 82025158, 82036781, 82036989, 82019471, 82017021, 82023560,
    82037779, 82028201, 82039898, 82015719, 82042777, 82009553, 82038860, 82042452, 82026787, 82036797,
    82030103, 82041598, 82022492, 82042991, 82003287, 82034349, 82037800, 82025069, 82025015, 82037835,
    82028370, 82042094, 82037880, 82042085, 82019447, 82041584, 82037823, 82023524, 82033417, 82042102,
    82038875, 82037816, 82011035, 82017050, 82042475, 82037818, 82026945, 82038832, 82042459, 82025061,
    82022225, 82029883, 82008543, 82028395, 82037195, 82028215, 82028331, 82018072, 82019559, 82034453,
    82038962, 82023517, 82039882, 82025121, 82026827, 82040772, 82042787, 82042176, 82031898
]

comid_orders_dict = {
    8: [82003287, 82025015, 82018084, 82008543, 82018072],
    7: [82036781, 82018092, 82023524, 82029872, 82028201],
    6: [82022185, 82023560, 82015719, 82029922, 82038832, 82026688, 82037779, 82037816, 82037800, 82040772, 82037818,
        82019447, 82037835, 82025069, 82035594, 82025057],
    5: [82023517, 82028215, 82031602, 82034349, 82039908, 82025121, 82041573, 82019471, 82019559, 82041584, 82028370,
        82038860, 82028307, 82022225, 82017050, 82008607, 82036797, 82036808],
    4: [82025061, 82005809, 82037880, 82017021, 82042777, 82037823, 82026827, 82026787, 82026945, 82038962, 82038875,
        82039917, 82039898, 82039882, 82030103, 82029883, 82042085, 82042094, 82042102, 82041599, 82041598, 82025133,
        82028331, 82025158, 82011128, 82028395, 82042452, 82042459, 82042445, 82022245, 82011035],
    3: [82014643, 82034453, 82036989, 82042475, 82042176, 82031898, 82022492, 82042991, 82033417, 82039999],
    2: [82028537, 82042787, 82037195],
    1: [82009553]
}

better_columns_orders = [
    'order_n-7_AirTemp', 'order_n-7_Albedo',
    'order_n-7_Avg_Skin_Temp', 'order_n-7_PlantCanopyWater',
    'order_n-7_CanopyWaterEvpn', 'order_n-7_DirectEvonBareSoil',
    'order_n-7_Evapotranspn', 'order_n-7_LngWaveRadFlux',
    'order_n-7_NetRadFlux', 'order_n-7_PotEvpnRate',
    'order_n-7_Pressure', 'order_n-7_SpecHmd', 'order_n-7_HeatFlux',
    'order_n-7_Sen.HtFlux', 'order_n-7_LtHeat',
    'order_n-7_StmSurfRunoff', 'order_n-7_BsGndWtrRunoff',
    'order_n-7_SnowMelt', 'order_n-7_TotalPcpRate',
    'order_n-7_RainPcpRate', 'order_n-7_RootZoneSoilMstr',
    'order_n-7_SnowDepthWtrEq', 'order_n-7_DwdShtWvRadFlux',
    'order_n-7_SnowDepth', 'order_n-7_SnowPcpRate',
    'order_n-7_SoilMst10', 'order_n-7_SoilMst40',
    'order_n-7_SoilMst100', 'order_n-7_SoilMst200',
    'order_n-7_SoilTmp10', 'order_n-7_SoilTmp40',
    'order_n-7_SoilTmp100', 'order_n-7_SoilTmp200',
    'order_n-7_NetShtWvRadFlux', 'order_n-7_Tspn', 'order_n-7_WindSpd',

    'order_n-6_AirTemp', 'order_n-6_Albedo',
    'order_n-6_Avg_Skin_Temp', 'order_n-6_PlantCanopyWater',
    'order_n-6_CanopyWaterEvpn', 'order_n-6_DirectEvonBareSoil',
    'order_n-6_Evapotranspn', 'order_n-6_LngWaveRadFlux',
    'order_n-6_NetRadFlux', 'order_n-6_PotEvpnRate',
    'order_n-6_Pressure', 'order_n-6_SpecHmd', 'order_n-6_HeatFlux',
    'order_n-6_Sen.HtFlux', 'order_n-6_LtHeat',
    'order_n-6_StmSurfRunoff', 'order_n-6_BsGndWtrRunoff',
    'order_n-6_SnowMelt', 'order_n-6_TotalPcpRate',
    'order_n-6_RainPcpRate', 'order_n-6_RootZoneSoilMstr',
    'order_n-6_SnowDepthWtrEq', 'order_n-6_DwdShtWvRadFlux',
    'order_n-6_SnowDepth', 'order_n-6_SnowPcpRate',
    'order_n-6_SoilMst10', 'order_n-6_SoilMst40',
    'order_n-6_SoilMst100', 'order_n-6_SoilMst200',
    'order_n-6_SoilTmp10', 'order_n-6_SoilTmp40',
    'order_n-6_SoilTmp100', 'order_n-6_SoilTmp200',
    'order_n-6_NetShtWvRadFlux', 'order_n-6_Tspn', 'order_n-6_WindSpd',

    'order_n-5_AirTemp', 'order_n-5_Albedo', 'order_n-5_Avg_Skin_Temp', 'order_n-5_PlantCanopyWater',
    'order_n-5_CanopyWaterEvpn', 'order_n-5_DirectEvonBareSoil',
    'order_n-5_Evapotranspn', 'order_n-5_LngWaveRadFlux', 'order_n-5_NetRadFlux',
    'order_n-5_PotEvpnRate', 'order_n-5_Pressure', 'order_n-5_SpecHmd', 'order_n-5_HeatFlux',
    'order_n-5_Sen.HtFlux', 'order_n-5_LtHeat', 'order_n-5_StmSurfRunoff',
    'order_n-5_BsGndWtrRunoff', 'order_n-5_SnowMelt', 'order_n-5_TotalPcpRate',
    'order_n-5_RainPcpRate', 'order_n-5_RootZoneSoilMstr', 'order_n-5_SnowDepthWtrEq',
    'order_n-5_DwdShtWvRadFlux', 'order_n-5_SnowDepth', 'order_n-5_SnowPcpRate',
    'order_n-5_SoilMst10', 'order_n-5_SoilMst40', 'order_n-5_SoilMst100',
    'order_n-5_SoilMst200', 'order_n-5_SoilTmp10', 'order_n-5_SoilTmp40',
    'order_n-5_SoilTmp100', 'order_n-5_SoilTmp200', 'order_n-5_NetShtWvRadFlux',
    'order_n-5_Tspn', 'order_n-5_WindSpd',

    'order_n-4_AirTemp', 'order_n-4_Albedo', 'order_n-4_Avg_Skin_Temp', 'order_n-4_PlantCanopyWater',
    'order_n-4_CanopyWaterEvpn',
    'order_n-4_DirectEvonBareSoil', 'order_n-4_Evapotranspn', 'order_n-4_LngWaveRadFlux',
    'order_n-4_NetRadFlux', 'order_n-4_PotEvpnRate', 'order_n-4_Pressure', 'order_n-4_SpecHmd',
    'order_n-4_HeatFlux', 'order_n-4_Sen.HtFlux', 'order_n-4_LtHeat',
    'order_n-4_StmSurfRunoff', 'order_n-4_BsGndWtrRunoff', 'order_n-4_SnowMelt',
    'order_n-4_TotalPcpRate', 'order_n-4_RainPcpRate', 'order_n-4_RootZoneSoilMstr',
    'order_n-4_SnowDepthWtrEq', 'order_n-4_DwdShtWvRadFlux', 'order_n-4_SnowDepth',
    'order_n-4_SnowPcpRate', 'order_n-4_SoilMst10', 'order_n-4_SoilMst40',
    'order_n-4_SoilMst100', 'order_n-4_SoilMst200', 'order_n-4_SoilTmp10',
    'order_n-4_SoilTmp40', 'order_n-4_SoilTmp100', 'order_n-4_SoilTmp200',
    'order_n-4_NetShtWvRadFlux', 'order_n-4_Tspn', 'order_n-4_WindSpd',

    'order_n-3_AirTemp', 'order_n-3_Albedo', 'order_n-3_Avg_Skin_Temp', 'order_n-3_PlantCanopyWater',
    'order_n-3_CanopyWaterEvpn', 'order_n-3_DirectEvonBareSoil',
    'order_n-3_Evapotranspn', 'order_n-3_LngWaveRadFlux', 'order_n-3_NetRadFlux',
    'order_n-3_PotEvpnRate', 'order_n-3_Pressure', 'order_n-3_SpecHmd', 'order_n-3_HeatFlux',
    'order_n-3_Sen.HtFlux', 'order_n-3_LtHeat', 'order_n-3_StmSurfRunoff',
    'order_n-3_BsGndWtrRunoff', 'order_n-3_SnowMelt', 'order_n-3_TotalPcpRate',
    'order_n-3_RainPcpRate', 'order_n-3_RootZoneSoilMstr', 'order_n-3_SnowDepthWtrEq',
    'order_n-3_DwdShtWvRadFlux', 'order_n-3_SnowDepth', 'order_n-3_SnowPcpRate',
    'order_n-3_SoilMst10', 'order_n-3_SoilMst40', 'order_n-3_SoilMst100',
    'order_n-3_SoilMst200', 'order_n-3_SoilTmp10', 'order_n-3_SoilTmp40',
    'order_n-3_SoilTmp100', 'order_n-3_SoilTmp200', 'order_n-3_NetShtWvRadFlux',
    'order_n-3_Tspn', 'order_n-3_WindSpd',

    'order_n-2_AirTemp', 'order_n-2_Albedo', 'order_n-2_Avg_Skin_Temp', 'order_n-2_PlantCanopyWater',
    'order_n-2_CanopyWaterEvpn',
    'order_n-2_DirectEvonBareSoil', 'order_n-2_Evapotranspn', 'order_n-2_LngWaveRadFlux',
    'order_n-2_NetRadFlux', 'order_n-2_PotEvpnRate', 'order_n-2_Pressure', 'order_n-2_SpecHmd',
    'order_n-2_HeatFlux', 'order_n-2_Sen.HtFlux', 'order_n-2_LtHeat',
    'order_n-2_StmSurfRunoff', 'order_n-2_BsGndWtrRunoff', 'order_n-2_SnowMelt',
    'order_n-2_TotalPcpRate', 'order_n-2_RainPcpRate', 'order_n-2_RootZoneSoilMstr',
    'order_n-2_SnowDepthWtrEq', 'order_n-2_DwdShtWvRadFlux', 'order_n-2_SnowDepth',
    'order_n-2_SnowPcpRate', 'order_n-2_SoilMst10', 'order_n-2_SoilMst40',
    'order_n-2_SoilMst100', 'order_n-2_SoilMst200', 'order_n-2_SoilTmp10',
    'order_n-2_SoilTmp40', 'order_n-2_SoilTmp100', 'order_n-2_SoilTmp200',
    'order_n-2_NetShtWvRadFlux', 'order_n-2_Tspn', 'order_n-2_WindSpd',

    'order_n-1_AirTemp', 'order_n-1_Albedo', 'order_n-1_Avg_Skin_Temp', 'order_n-1_PlantCanopyWater',
    'order_n-1_CanopyWaterEvpn', 'order_n-1_DirectEvonBareSoil',
    'order_n-1_Evapotranspn', 'order_n-1_LngWaveRadFlux', 'order_n-1_NetRadFlux',
    'order_n-1_PotEvpnRate', 'order_n-1_Pressure', 'order_n-1_SpecHmd', 'order_n-1_HeatFlux',
    'order_n-1_Sen.HtFlux', 'order_n-1_LtHeat', 'order_n-1_StmSurfRunoff',
    'order_n-1_BsGndWtrRunoff', 'order_n-1_SnowMelt', 'order_n-1_TotalPcpRate',
    'order_n-1_RainPcpRate', 'order_n-1_RootZoneSoilMstr', 'order_n-1_SnowDepthWtrEq',
    'order_n-1_DwdShtWvRadFlux', 'order_n-1_SnowDepth', 'order_n-1_SnowPcpRate',
    'order_n-1_SoilMst10', 'order_n-1_SoilMst40', 'order_n-1_SoilMst100',
    'order_n-1_SoilMst200', 'order_n-1_SoilTmp10', 'order_n-1_SoilTmp40',
    'order_n-1_SoilTmp100', 'order_n-1_SoilTmp200', 'order_n-1_NetShtWvRadFlux',
    'order_n-1_Tspn', 'order_n-1_WindSpd',

    'order_n_AirTemp', 'order_n_Albedo', 'order_n_Avg_Skin_Temp', 'order_n_PlantCanopyWater', 'order_n_CanopyWaterEvpn',
    'order_n_DirectEvonBareSoil', 'order_n_Evapotranspn', 'order_n_LngWaveRadFlux',
    'order_n_NetRadFlux', 'order_n_PotEvpnRate', 'order_n_Pressure', 'order_n_SpecHmd',
    'order_n_HeatFlux', 'order_n_Sen.HtFlux', 'order_n_LtHeat', 'order_n_StmSurfRunoff',
    'order_n_BsGndWtrRunoff', 'order_n_SnowMelt', 'order_n_TotalPcpRate',
    'order_n_RainPcpRate', 'order_n_RootZoneSoilMstr', 'order_n_SnowDepthWtrEq',
    'order_n_DwdShtWvRadFlux', 'order_n_SnowDepth', 'order_n_SnowPcpRate', 'order_n_SoilMst10',
    'order_n_SoilMst40', 'order_n_SoilMst100', 'order_n_SoilMst200', 'order_n_SoilTmp10',
    'order_n_SoilTmp40', 'order_n_SoilTmp100', 'order_n_SoilTmp200', 'order_n_NetShtWvRadFlux',
    'order_n_Tspn', 'order_n_WindSpd',

    'Q', 'Albedo', 'Avg_Skin_Temp', 'PlantCanopyWater', 'CanopyWaterEvpn', 'DirectEvonBareSoil', 'Evapotranspn',
    'LngWaveRadFlux', 'NetRadFlux', 'PotEvpnRate', 'Pressure', 'SpecHmd', 'HeatFlux', 'Sen.HtFlux', 'LtHeat',
    'StmSurfRunoff',
    'BsGndWtrRunoff', 'SnowMelt', 'TotalPcpRate', 'RainPcpRate', 'RootZoneSoilMstr', 'SnowDepthWtrEq',
    'DwdShtWvRadFlux',
    'SnowDepth', 'SnowPcpRate', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp10', 'SoilTmp40',
    'SoilTmp100', 'SoilTmp200',
    'NetShtWvRadFlux', 'AirTemp', 'Tspn', 'WindSpd', 'NDVI',

    'lengthkm', 'sinuosity', 'slope', 'uparea', 'width_max', 'lengthdir', 'width_mean',
    'width',
    'order',
    'discharge',
]

static_features = [
    'lengthkm', 'sinuosity', 'slope', 'uparea', 'width_max', 'lengthdir', 'width_mean',
]
categorical_features = ['order']

columns_to_scale = [item for item in better_columns_orders if item not in categorical_features]

columns_to_drop = ['CanopyWaterEvpn', 'DirectEvonBareSoil', 'LngWaveRadFlux', 'HeatFlux', 'StmSurfRunoff', 'BsGndWtrRunoff', 'RainPcpRate',
                   'RootZoneSoilMstr', 'DwdShtWvRadFlux', 'SnowDepth', 'SoilMst10', 'SoilMst40', 'SoilMst100', 'SoilMst200', 'SoilTmp40', 'SoilTmp100', 'SoilTmp200']

"""
# ============================================= ORDERS 4 TO 8 ==========================================================
nchoose_10_sets = [
    [82010991, 82015719, 82019471, 82042459, 82019559, 82008543, 82018092, 82036797, 82018084, 82029922],
    [82012264, 82015719, 82041599, 82042445, 82025015, 82019447, 82026827, 82018092, 82038860, 82042777],
    [82014701, 82041573, 82038962, 82017050, 82037823, 82025069, 82036781, 82042094, 82035594, 82028395],
    [82017021, 82028307, 82017050, 82041599, 82039898, 82037835, 82041584, 82023517, 82017051, 82037816],
    [82017050, 82026787, 82028395, 82014701, 82023560, 82036781, 82022185, 82026945, 82030103, 82019471],
    [82017050, 82041598, 82008688, 82028395, 82030103, 82042094, 82042777, 82004876, 82003287, 82017021],
    [82017051, 82041573, 82041599, 82037818, 82023524, 82037880, 82042777, 82026827, 82034349, 82037835],
    [82018084, 82026945, 82028331, 82008543, 82026688, 82025061, 82029883, 82010991, 82023560, 82041598],
    [82018084, 82037818, 82036797, 82040772, 82039882, 82025015, 82014602, 82042085, 82029883, 82037816],
    [82019471, 82038860, 82004876, 82008543, 82036797, 82026688, 82039908, 82028307, 82022245, 82037880],
    [82019559, 82025015, 82038875, 82037880, 82008543, 82037835, 82039917, 82042452, 82042777, 82017051],
    [82022185, 82042102, 82042445, 82025158, 82003287, 82042777, 82028331, 82037835, 82036808, 82036781],
    [82023517, 82031602, 82017050, 82037823, 82028395, 82025015, 82028201, 82040772, 82035594, 82037779],
    [82025061, 82025015, 82015719, 82042445, 82019471, 82038875, 82041584, 82022185, 82037835, 82019447],
    [82025061, 82025121, 82028307, 82025069, 82028395, 82041598, 82008607, 82042102, 82017021, 82004876],
    [82025069, 82036808, 82025121, 82037818, 82039882, 82039917, 82028331, 82028201, 82029922, 82011035],
    [82026688, 82025158, 82022185, 82018072, 82026787, 82025061, 82042452, 82041584, 82008607, 82037800],
    [82028331, 82017050, 82029922, 82019471, 82028201, 82038875, 82039908, 82003287, 82042452, 82011128],
    [82029883, 82042777, 82036781, 82025069, 82039898, 82023524, 82025057, 82003287, 82019471, 82014602],
    [82031602, 82010991, 82019559, 82036808, 82019471, 82017051, 82018092, 82028395, 82025057, 82041573],
    [82034349, 82025133, 82025121, 82028215, 82042445, 82022245, 82008543, 82005809, 82023560, 82010991],
    [82036781, 82037800, 82028215, 82042094, 82019447, 82025061, 82022245, 82039882, 82025158, 82003287],
    [82037779, 82022185, 82042777, 82017050, 82025015, 82036781, 82037823, 82029883, 82042085, 82029922],
    [82038832, 82028201, 82040772, 82008607, 82041584, 82039908, 82037818, 82038860, 82015719, 82028215],
    [82038860, 82011128, 82028215, 82022245, 82026945, 82025133, 82036781, 82018072, 82040772, 82038962],
    [82041598, 82023560, 82025069, 82003287, 82015719, 82019471, 82031602, 82042445, 82014602, 82028395],
    [82042459, 82012264, 82014701, 82026827, 82037818, 82041599, 82042094, 82031602, 82023517, 82030103],
    [82042459, 82037835, 82019559, 82005809, 82028201, 82042085, 82025133, 82034349, 82019471, 82017051],
    [82042777, 82028331, 82010991, 82005809, 82036797, 82008607, 82029922, 82041598, 82025057, 82014602],
    [82042777, 82038875, 82025133, 82042445, 82010991, 82042094, 82038962, 82023560, 82017021, 82039898]
]
# =================================================================================================================================================
"""
# ========================================ORDERS 2 TO 8 =======================================
nchoose_10_sets = [
    [82010991, 82015719, 82019471, 82042459, 82019559, 82008543, 82018092, 82036797, 82018084, 82029922],
    [82012264, 82015719, 82041599, 82042445, 82025015, 82019447, 82026827, 82018092, 82038860, 82042777],
    [82014701, 82041573, 82038962, 82017050, 82037823, 82025069, 82036781, 82042094, 82035594, 82028395],
    [82017021, 82028307, 82017050, 82041599, 82039898, 82037835, 82041584, 82023517, 82017051, 82037816],
    [82017050, 82026787, 82028395, 82014701, 82023560, 82036781, 82022185, 82026945, 82030103, 82019471],
    [82017050, 82041598, 82008688, 82028395, 82030103, 82042094, 82042777, 82004876, 82003287, 82017021],
    [82017051, 82041573, 82041599, 82037818, 82023524, 82037880, 82042777, 82026827, 82034349, 82037835],
    [82018084, 82026945, 82028331, 82008543, 82026688, 82025061, 82029883, 82010991, 82023560, 82041598],
    [82018084, 82037818, 82036797, 82040772, 82039882, 82025015, 82014602, 82042085, 82029883, 82037816],
    [82019471, 82038860, 82004876, 82008543, 82036797, 82026688, 82039908, 82028307, 82022245, 82037880],
    [82019559, 82025015, 82038875, 82037880, 82008543, 82037835, 82039917, 82042452, 82042777, 82017051],
    [82022185, 82042102, 82042445, 82025158, 82003287, 82042777, 82028331, 82037835, 82036808, 82036781],
    [82023517, 82031602, 82017050, 82037823, 82028395, 82025015, 82028201, 82040772, 82035594, 82037779],
    [82025061, 82025015, 82015719, 82042445, 82019471, 82038875, 82041584, 82022185, 82037835, 82019447],
    [82025061, 82025121, 82028307, 82025069, 82028395, 82041598, 82008607, 82042102, 82017021, 82004876],
    [82025069, 82036808, 82025121, 82037818, 82039882, 82039917, 82028331, 82028201, 82029922, 82011035],
    [82026688, 82025158, 82022185, 82018072, 82026787, 82025061, 82042452, 82041584, 82008607, 82037800],
    [82028331, 82017050, 82029922, 82019471, 82028201, 82038875, 82039908, 82003287, 82042452, 82011128],
    [82029883, 82042777, 82036781, 82025069, 82039898, 82023524, 82025057, 82003287, 82019471, 82014602],
    [82031602, 82010991, 82019559, 82036808, 82019471, 82017051, 82018092, 82028395, 82025057, 82041573],
    [82034349, 82025133, 82025121, 82028215, 82042445, 82022245, 82008543, 82005809, 82023560, 82010991],
    [82036781, 82037800, 82028215, 82042094, 82019447, 82025061, 82022245, 82039882, 82025158, 82003287],
    [82014643, 82031898, 82042991, 82033417, 82037195, 82022492, 82042176, 82042475, 82036989, 82034453],
    [82014643, 82042475, 82031898, 82022492, 82042991, 82039999, 82037195, 82028537, 82036989, 82042787],
    [82022492, 82014643, 82036989, 82042787, 82031898, 82039999, 82037195, 82042475, 82042176, 82042991],
    [82022492, 82033417, 82034453, 82028537, 82039999, 82031898, 82042991, 82036989, 82042787, 82042176],
    [82022492, 82039999, 82042176, 82028537, 82036989, 82042991, 82031898, 82034453, 82042475, 82042787],
    [82028537, 82031898, 82033417, 82014643, 82042475, 82042176, 82022492, 82039999, 82042787, 82042991],
    [82028537, 82033417, 82014643, 82034453, 82039999, 82022492, 82042991, 82042475, 82042176, 82037195],
    [82028537, 82034453, 82014643, 82037195, 82039999, 82042176, 82022492, 82042475, 82031898, 82042787],
]

nchoose_20_sets = [
    [82011128, 82008607, 82022245, 82028307, 82019447, 82028201, 82017050, 82037880, 82028215, 82039898, 82023517, 82038860, 82029922, 82037816, 82008543, 82005809, 82042094, 82041599, 82028395, 82025121],
    [82015719, 82039917, 82028395, 82038962, 82037779, 82028215, 82036781, 82005809, 82029872, 82037818, 82019447, 82041573, 82039882, 82034349, 82018092, 82022225, 82042094, 82029883, 82041584, 82038832],
    [82017050, 82023560, 82028201, 82011128, 82030103, 82023524, 82028307, 82017021, 82029872, 82026827, 82018072, 82042102, 82042094, 82042445, 82037816, 82018092, 82025158, 82025061, 82039882, 82036808],
    [82018084, 82011035, 82022225, 82023560, 82042094, 82039898, 82035594, 82028307, 82041598, 82025015, 82037880, 82042459, 82025158, 82036781, 82011128, 82018092, 82041573, 82038832, 82026827, 82019559],
    [82019471, 82018092, 82037816, 82040772, 82037800, 82025133, 82025061, 82026945, 82025158, 82036781, 82028215, 82041573, 82008543, 82031602, 82042459, 82034349, 82023517, 82039908, 82036797, 82041584],
    [82019559, 82026945, 82039917, 82035594, 82023524, 82031602, 82041584, 82038875, 82038832, 82041573, 82029883, 82017050, 82019471, 82036808, 82025015, 82025069, 82023560, 82018084, 82040772, 82022225],
    [82025069, 82042445, 82018072, 82038860, 82028370, 82036808, 82026827, 82036781, 82026688, 82034349, 82037835, 82025015, 82029883, 82019559, 82041584, 82042102, 82025057, 82015719, 82028215, 82037779],
    [82026827, 82008543, 82025061, 82028395, 82025057, 82022225, 82028215, 82008607, 82042459, 82042777, 82042445, 82042085, 82029883, 82041584, 82031602, 82019559, 82026945, 82037779, 82041573, 82017050],
    [82028215, 82008607, 82025158, 82041598, 82029922, 82005809, 82029883, 82028331, 82028307, 82034349, 82042102, 82018092, 82037823, 82022185, 82039908, 82042459, 82038875, 82037818, 82025057, 82037779],
    [82028215, 82039882, 82036808, 82037800, 82025121, 82038832, 82040772, 82041584, 82028395, 82028331, 82026688, 82030103, 82019471, 82037818, 82017021, 82037835, 82025133, 82022245, 82028370, 82035594],
    [82028307, 82037823, 82029922, 82025133, 82028331, 82022245, 82037880, 82011128, 82038962, 82026945, 82008607, 82017021, 82017050, 82042102, 82028370, 82029883, 82028201, 82040772, 82025121, 82042777],
    [82029872, 82018092, 82037818, 82028307, 82041573, 82025015, 82008543, 82031602, 82026945, 82039908, 82035594, 82011128, 82015719, 82023517, 82042452, 82028215, 82025061, 82037835, 82022245, 82037880],
    [82029872, 82025061, 82037835, 82036781, 82042452, 82026945, 82037818, 82037880, 82028331, 82025015, 82025069, 82008607, 82023524, 82026787, 82039898, 82019447, 82030103, 82018092, 82037816, 82041573],
    [82035594, 82039917, 82026787, 82041598, 82041599, 82028201, 82026688, 82037818, 82041584, 82041573, 82025121, 82023524, 82031602, 82011035, 82029922, 82022245, 82028370, 82028331, 82025133, 82040772],
    [82036808, 82037880, 82025061, 82042085, 82036797, 82038875, 82018092, 82042777, 82019447, 82025121, 82029883, 82037816, 82040772, 82025057, 82023517, 82037835, 82022225, 82042445, 82023524, 82011035],
    [82036808, 82042777, 82035594, 82031602, 82039882, 82041598, 82026945, 82042445, 82041584, 82017050, 82029872, 82028395, 82011035, 82008607, 82040772, 82011128, 82042102, 82023524, 82025061, 82003287],
    [82037800, 82037818, 82036781, 82018092, 82039882, 82023524, 82039898, 82023517, 82023560, 82029922, 82037779, 82025015, 82025158, 82029883, 82042777, 82041573, 82036808, 82028395, 82039908, 82042094],
    [82037818, 82037816, 82022225, 82008543, 82028215, 82034349, 82025015, 82030103, 82042094, 82017021, 82005809, 82036797, 82025057, 82019559, 82042102, 82015719, 82042777, 82037779, 82028370, 82042452],
    [82037823, 82025158, 82039898, 82034349, 82039882, 82038860, 82036781, 82029872, 82028331, 82028370, 82041573, 82018092, 82026688, 82042102, 82028395, 82037779, 82039917, 82040772, 82041599, 82041598],
    [82037823, 82028395, 82039917, 82029872, 82022185, 82037779, 82042085, 82029922, 82036781, 82025057, 82022225, 82026688, 82042459, 82037835, 82005809, 82019471, 82023524, 82031602, 82039908, 82011128],
    [82037823, 82040772, 82041598, 82011128, 82023517, 82028395, 82028215, 82038962, 82025015, 82008607, 82031602, 82025133, 82025158, 82026688, 82038832, 82023524, 82029883, 82036797, 82042085, 82037800],
    [82038832, 82030103, 82041599, 82008607, 82025015, 82018084, 82025061, 82025057, 82034349, 82041584, 82037779, 82037800, 82019471, 82008543, 82028307, 82038962, 82011035, 82038875, 82026787, 82011128],
    [82038860, 82023560, 82042452, 82011035, 82039882, 82019471, 82030103, 82041573, 82031602, 82036808, 82005809, 82036781, 82028307, 82041584, 82037779, 82025057, 82019447, 82026945, 82022185, 82029872],
    [82039908, 82019447, 82011128, 82017050, 82042459, 82025133, 82042777, 82028395, 82008543, 82037823, 82042445, 82038860, 82037835, 82041584, 82015719, 82026787, 82019559, 82023517, 82042085, 82036797],
    [82041573, 82026945, 82022245, 82028395, 82028331, 82018092, 82037818, 82041598, 82026688, 82023524, 82039908, 82038860, 82042445, 82025121, 82011128, 82042777, 82025133, 82023517, 82017050, 82025069],
    [82042094, 82023517, 82026945, 82017021, 82011035, 82022225, 82039898, 82042452, 82028395, 82038832, 82038875, 82025015, 82018092, 82037800, 82011128, 82028331, 82018072, 82025057, 82041573, 82026827],
    [82042102, 82042094, 82037800, 82005809, 82037818, 82018072, 82022185, 82017050, 82037880, 82042085, 82040772, 82030103, 82041599, 82026787, 82023560, 82018092, 82026688, 82025057, 82035594, 82017021],
    [82042452, 82022225, 82028201, 82028331, 82008543, 82039917, 82037779, 82025015, 82018072, 82019447, 82022245, 82029883, 82028395, 82028215, 82038832, 82026945, 82003287, 82038962, 82025158, 82017050],
    [82042459, 82039882, 82026945, 82041573, 82037816, 82042102, 82011128, 82042094, 82028370, 82041598, 82039917, 82042445, 82028201, 82038875, 82029922, 82019559, 82018084, 82023524, 82026688, 82023517],
    [82042459, 82042445, 82026945, 82042777, 82038875, 82041598, 82017050, 82029872, 82028370, 82025069, 82031602, 82042102, 82025121, 82029883, 82008543, 82025158, 82034349, 82011035, 82037779, 82038860]
]
stations_sets_dict = {
    10: nchoose_10_sets,
    20: nchoose_20_sets,
}

# NEW SETS OF 3 FOR TESTING ....


orders_to_drop = {
    7: ['n-7', 'n-6', 'n-5', 'n-4', 'n-3', 'n-2', 'n-1'],
    6: ['n-7', 'n-6', 'n-5', 'n-4', 'n-3', 'n-2'],
    5: ['n-7', 'n-6', 'n-5', 'n-4', 'n-3'],
    4: ['n-7', 'n-6', 'n-5', 'n-4'],
    3: ['n-7', 'n-6', 'n-5'],
    2: ['n-7', 'n-6'],
    1: ['n-7'],
    0: [],
}