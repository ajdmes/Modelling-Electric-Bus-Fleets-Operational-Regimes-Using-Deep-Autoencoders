#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 18:37:26 2026

@author: alexandresilva
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
#warnings.filterwarnings("ignore")


path = './data/data_bus478/'

# read all feature files, concatenate them and merge them into a single dataframe

hd_volt_1 = pd.read_parquet(path + 'datahub_hardware.volt_2025-12-01.parquet')
hd_volt_2 = pd.read_parquet(path + 'datahub_hardware.volt_2026-01-01.parquet')
hd_volt_3 = pd.read_parquet(path + 'datahub_hardware.volt_2026-02-01.parquet')
hd_volt_4 = pd.read_parquet(path + 'datahub_hardware.volt_2026-02-18.parquet')
hd_volt = pd.concat([hd_volt_1, hd_volt_2, hd_volt_3, hd_volt_4], ignore_index=True, sort=False)


ign_volt_1 = pd.read_parquet(path + 'datahub.ignition_voltage_2025-12-01.parquet')
ign_volt_2 = pd.read_parquet(path + 'datahub.ignition_voltage_2026-01-01.parquet')
ign_volt_3 = pd.read_parquet(path + 'datahub.ignition_voltage_2026-02-01.parquet')
ign_volt_4 = pd.read_parquet(path + 'datahub.ignition_voltage_2026-02-18.parquet')
ign_volt = pd.concat([ign_volt_1, ign_volt_2, ign_volt_3, ign_volt_4], ignore_index=True, sort=False)


gfd_res_1 = pd.read_parquet(path + 's01_ecu_comms_evcuccu_errors1.gfd_resistance_2025-12-01.parquet')
gfd_res_2 = pd.read_parquet(path + 's01_ecu_comms_evcuccu_errors1.gfd_resistance_2026-01-01.parquet')
gfd_res_3 = pd.read_parquet(path + 's01_ecu_comms_evcuccu_errors1.gfd_resistance_2026-02-01.parquet')
gfd_res_4 = pd.read_parquet(path + 's01_ecu_comms_evcuccu_errors1.gfd_resistance_2026-02-18.parquet')
gfd_res = pd.concat([gfd_res_1, gfd_res_2, gfd_res_3, gfd_res_4], ignore_index=True, sort=False)


sys_phase_1 = pd.read_parquet(path + 's01_ecu_comms_vehicle_status.vehicle_system_phase_2025-12-01.parquet')
sys_phase_2 = pd.read_parquet(path + 's01_ecu_comms_vehicle_status.vehicle_system_phase_2026-01-01.parquet')
sys_phase_3 = pd.read_parquet(path + 's01_ecu_comms_vehicle_status.vehicle_system_phase_2026-02-01.parquet')
sys_phase_4 = pd.read_parquet(path + 's01_ecu_comms_vehicle_status.vehicle_system_phase_2026-02-18.parquet')
sys_phase = pd.concat([sys_phase_1, sys_phase_2, sys_phase_3, sys_phase_4], ignore_index=True, sort=False)


bat_curr_set_1 = pd.read_parquet(path + 's03_bms_b2_v_charge_info.b2_c_s_current_set_2025-12-01.parquet')
bat_curr_set_2 = pd.read_parquet(path + 's03_bms_b2_v_charge_info.b2_c_s_current_set_2026-01-01.parquet')
bat_curr_set_3 = pd.read_parquet(path + 's03_bms_b2_v_charge_info.b2_c_s_current_set_2026-02-01.parquet')
bat_curr_set_4 = pd.read_parquet(path + 's03_bms_b2_v_charge_info.b2_c_s_current_set_2026-02-18.parquet')
bat_curr_set = pd.concat([bat_curr_set_1, bat_curr_set_2, bat_curr_set_3, bat_curr_set_4], ignore_index=True, sort=False)


bat_volt_set_1 = pd.read_parquet(path + 's03_bms_b2_v_charge_info.b2_c_s_voltage_set_2025-12-01.parquet')
bat_volt_set_2 = pd.read_parquet(path + 's03_bms_b2_v_charge_info.b2_c_s_voltage_set_2026-01-01.parquet')
bat_volt_set_3 = pd.read_parquet(path + 's03_bms_b2_v_charge_info.b2_c_s_voltage_set_2026-02-01.parquet')
bat_volt_set_4 = pd.read_parquet(path + 's03_bms_b2_v_charge_info.b2_c_s_voltage_set_2026-02-18.parquet')
bat_volt_set = pd.concat([bat_volt_set_1, bat_volt_set_2, bat_volt_set_3, bat_volt_set_4], ignore_index=True, sort=False)


bat_soc_1 = pd.read_parquet(path + 's03_bms_b2_v_st2.b2_v_st2_soc_2025-12-01.parquet')
bat_soc_2 = pd.read_parquet(path + 's03_bms_b2_v_st2.b2_v_st2_soc_2026-01-01.parquet')
bat_soc_3 = pd.read_parquet(path + 's03_bms_b2_v_st2.b2_v_st2_soc_2026-02-01.parquet')
bat_soc_4 = pd.read_parquet(path + 's03_bms_b2_v_st2.b2_v_st2_soc_2026-02-18.parquet')
bat_soc = pd.concat([bat_soc_1, bat_soc_2, bat_soc_3, bat_soc_4], ignore_index=True, sort=False)


aux_wp_1 = pd.read_parquet(path + 's05_cooling_auxiliary_wp_cmd.rpm_speed_cmd_2025-12-01.parquet')
aux_wp_2 = pd.read_parquet(path + 's05_cooling_auxiliary_wp_cmd.rpm_speed_cmd_2026-01-01.parquet')
aux_wp_3 = pd.read_parquet(path + 's05_cooling_auxiliary_wp_cmd.rpm_speed_cmd_2026-02-01.parquet')
aux_wp_4 = pd.read_parquet(path + 's05_cooling_auxiliary_wp_cmd.rpm_speed_cmd_2026-02-18.parquet')
aux_wp = pd.concat([aux_wp_1, aux_wp_2, aux_wp_3, aux_wp_4], ignore_index=True, sort=False)


aux_cw_temp_1 = pd.read_parquet(path + 's05_cooling_analysis.cold_water_temp_auxiliar_2025-12-01.parquet')
aux_cw_temp_2 = pd.read_parquet(path + 's05_cooling_analysis.cold_water_temp_auxiliar_2026-01-01.parquet')
aux_cw_temp_3 = pd.read_parquet(path + 's05_cooling_analysis.cold_water_temp_auxiliar_2026-02-01.parquet')
aux_cw_temp_4 = pd.read_parquet(path + 's05_cooling_analysis.cold_water_temp_auxiliar_2026-02-18.parquet')
aux_cw_temp = pd.concat([aux_cw_temp_1, aux_cw_temp_2, aux_cw_temp_3, aux_cw_temp_4], ignore_index=True, sort=False)


aux_hw_temp_1 = pd.read_parquet(path + 's05_cooling_analysis.hot_water_temp_auxiliar_2025-12-01.parquet')
aux_hw_temp_2 = pd.read_parquet(path + 's05_cooling_analysis.hot_water_temp_auxiliar_2026-01-01.parquet')
aux_hw_temp_3 = pd.read_parquet(path + 's05_cooling_analysis.hot_water_temp_auxiliar_2026-02-01.parquet')
aux_hw_temp_4 = pd.read_parquet(path + 's05_cooling_analysis.hot_water_temp_auxiliar_2026-02-18.parquet')
aux_hw_temp = pd.concat([aux_hw_temp_1, aux_hw_temp_2, aux_hw_temp_3, aux_hw_temp_4], ignore_index=True, sort=False)


trac_wp_1 = pd.read_parquet(path + 's05_cooling_traction_wp_cmd.rpm_speed_cmd_2025-12-01.parquet')
trac_wp_2 = pd.read_parquet(path + 's05_cooling_traction_wp_cmd.rpm_speed_cmd_2026-01-01.parquet')
trac_wp_3 = pd.read_parquet(path + 's05_cooling_traction_wp_cmd.rpm_speed_cmd_2026-02-01.parquet')
trac_wp_4 = pd.read_parquet(path + 's05_cooling_traction_wp_cmd.rpm_speed_cmd_2026-02-18.parquet')
trac_wp = pd.concat([trac_wp_1, trac_wp_2, trac_wp_3, trac_wp_4], ignore_index=True, sort=False)


trac_cw_temp_1 = pd.read_parquet(path + 's05_cooling_analysis.cold_water_temp_traction_2025-12-01.parquet')
trac_cw_temp_2 = pd.read_parquet(path + 's05_cooling_analysis.cold_water_temp_traction_2026-01-01.parquet')
trac_cw_temp_3 = pd.read_parquet(path + 's05_cooling_analysis.cold_water_temp_traction_2026-02-01.parquet')
trac_cw_temp_4 = pd.read_parquet(path + 's05_cooling_analysis.cold_water_temp_traction_2026-02-18.parquet')
trac_cw_temp = pd.concat([trac_cw_temp_1, trac_cw_temp_2, trac_cw_temp_3, trac_cw_temp_4], ignore_index=True, sort=False)


trac_hw_temp_1 = pd.read_parquet(path + 's05_cooling_analysis.hot_water_temp_traction_2025-12-01.parquet')
trac_hw_temp_2 = pd.read_parquet(path + 's05_cooling_analysis.hot_water_temp_traction_2026-01-01.parquet')
trac_hw_temp_3 = pd.read_parquet(path + 's05_cooling_analysis.hot_water_temp_traction_2026-02-01.parquet')
trac_hw_temp_4 = pd.read_parquet(path + 's05_cooling_analysis.hot_water_temp_traction_2026-02-18.parquet')
trac_hw_temp = pd.concat([trac_hw_temp_1, trac_hw_temp_2, trac_hw_temp_3, trac_hw_temp_4], ignore_index=True, sort=False)


aircomp_speed_1 = pd.read_parquet(path + 's06_aircomp_status_hs1_torque.speed_measured_2025-12-01.parquet')
aircomp_speed_2 = pd.read_parquet(path + 's06_aircomp_status_hs1_torque.speed_measured_2026-01-01.parquet')
aircomp_speed_3 = pd.read_parquet(path + 's06_aircomp_status_hs1_torque.speed_measured_2026-02-01.parquet')
aircomp_speed_4 = pd.read_parquet(path + 's06_aircomp_status_hs1_torque.speed_measured_2026-02-18.parquet')
aircomp_speed = pd.concat([aircomp_speed_1, aircomp_speed_2, aircomp_speed_3, aircomp_speed_4], ignore_index=True, sort=False)


aircomp_temp_1 = pd.read_parquet(path + 's06_aircomp_status_hs3_temp.motor_t_emperature_2025-12-01.parquet')
aircomp_temp_2 = pd.read_parquet(path + 's06_aircomp_status_hs3_temp.motor_t_emperature_2026-01-01.parquet')
aircomp_temp_3 = pd.read_parquet(path + 's06_aircomp_status_hs3_temp.motor_t_emperature_2026-02-01.parquet')
aircomp_temp_4 = pd.read_parquet(path + 's06_aircomp_status_hs3_temp.motor_t_emperature_2026-02-18.parquet')
aircomp_temp = pd.concat([aircomp_temp_1, aircomp_temp_2, aircomp_temp_3, aircomp_temp_4], ignore_index=True, sort=False)


dcdc1_inv_stat_1 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_inverter_status_2025-12-01.parquet')
dcdc1_inv_stat_2 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_inverter_status_2026-01-01.parquet')
dcdc1_inv_stat_3 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_inverter_status_2026-02-01.parquet')
dcdc1_inv_stat_4 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_inverter_status_2026-02-18.parquet')
dcdc1_inv_stat = pd.concat([dcdc1_inv_stat_1, dcdc1_inv_stat_2, dcdc1_inv_stat_3, dcdc1_inv_stat_4], ignore_index=True, sort=False)


dcdc1_power_1 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_power_2025-12-01.parquet')
dcdc1_power_2 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_power_2026-01-01.parquet')
dcdc1_power_3 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_power_2026-02-01.parquet')
dcdc1_power_4 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_power_2026-02-18.parquet')
dcdc1_power = pd.concat([dcdc1_power_1, dcdc1_power_2, dcdc1_power_3, dcdc1_power_4], ignore_index=True, sort=False)


dcdc1_temp_1 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_temperature_2025-12-01.parquet')
dcdc1_temp_2 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_temperature_2026-01-01.parquet')
dcdc1_temp_3 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_temperature_2026-02-01.parquet')
dcdc1_temp_4 = pd.read_parquet(path + 's10_dcdc1_actual_values.dcdc1_actual_temperature_2026-02-18.parquet')
dcdc1_temp = pd.concat([dcdc1_temp_1, dcdc1_temp_2, dcdc1_temp_3, dcdc1_temp_4], ignore_index=True, sort=False)


dcdc2_inv_stat_1 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_inverter_status_2025-12-01.parquet')
dcdc2_inv_stat_2 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_inverter_status_2026-01-01.parquet')
dcdc2_inv_stat_3 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_inverter_status_2026-02-01.parquet')
dcdc2_inv_stat_4 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_inverter_status_2026-02-18.parquet')
dcdc2_inv_stat = pd.concat([dcdc2_inv_stat_1, dcdc2_inv_stat_2, dcdc2_inv_stat_3, dcdc2_inv_stat_4], ignore_index=True, sort=False)


dcdc2_power_1 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_actual_power_2025-12-01.parquet')
dcdc2_power_2 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_actual_power_2026-01-01.parquet')
dcdc2_power_3 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_actual_power_2026-02-01.parquet')
dcdc2_power_4 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_actual_power_2026-02-18.parquet')
dcdc2_power = pd.concat([dcdc2_power_1, dcdc2_power_2, dcdc2_power_3, dcdc2_power_4], ignore_index=True, sort=False)


dcdc2_temp_1 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_actual_temperature_2025-12-01.parquet')
dcdc2_temp_2 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_actual_temperature_2026-01-01.parquet')
dcdc2_temp_3 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_actual_temperature_2026-02-01.parquet')
dcdc2_temp_4 = pd.read_parquet(path + 's10_dcdc2_actual_values.dcdc2_actual_temperature_2026-02-18.parquet')
dcdc2_temp = pd.concat([dcdc2_temp_1, dcdc2_temp_2, dcdc2_temp_3, dcdc2_temp_4], ignore_index=True, sort=False)


dcac3_inv_stat_1 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_atual_inverter_status_2025-12-01.parquet')
dcac3_inv_stat_2 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_atual_inverter_status_2026-01-01.parquet')
dcac3_inv_stat_3 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_atual_inverter_status_2026-02-01.parquet')
dcac3_inv_stat_4 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_atual_inverter_status_2026-02-18.parquet')
dcac3_inv_stat = pd.concat([dcac3_inv_stat_1, dcac3_inv_stat_2, dcac3_inv_stat_3, dcac3_inv_stat_4], ignore_index=True, sort=False)


dcac3_power_1 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_actual_power_2025-12-01.parquet')
dcac3_power_2 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_actual_power_2026-01-01.parquet')
dcac3_power_3 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_actual_power_2026-02-01.parquet')
dcac3_power_4 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_actual_power_2026-02-18.parquet')
dcac3_power = pd.concat([dcac3_power_1, dcac3_power_2, dcac3_power_3, dcac3_power_4], ignore_index=True, sort=False)


dcac3_temp_1 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_actual_temperature_2025-12-01.parquet')
dcac3_temp_2 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_actual_temperature_2026-01-01.parquet')
dcac3_temp_3 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_actual_temperature_2026-02-01.parquet')
dcac3_temp_4 = pd.read_parquet(path + 's10_dcac3_actual_values.dcdc3_hvac_actual_temperature_2026-02-18.parquet')
dcac3_temp = pd.concat([dcac3_temp_1, dcac3_temp_2, dcac3_temp_3, dcac3_temp_4], ignore_index=True, sort=False)


features = ['hd_volt',
            'ign_volt',
            'gfd_res',
            'sys_phase',
            'bat_curr_set',
            'bat_volt_set',
            'bat_soc',
            'aux_wp',
            'aux_cw_temp',
            'aux_hw_temp',
            'trac_wp',
            'trac_cw_temp',
            'trac_hw_temp',
            'aircomp_speed',
            'aircomp_temp',
            'dcdc1_inv_stat',
            'dcdc1_power',
            'dcdc1_temp',
            'dcdc2_inv_stat',
            'dcdc2_power',
            'dcdc2_temp',
            'dcac3_inv_stat',
            'dcac3_power',
            'dcac3_temp']


# check non informative features and possible categorical ones

for var in features:
    
    print(var, ':', eval(var).value.nunique())
    
    
# remove non informative features

features.remove('hd_volt')
features.remove('ign_volt')
features.remove('bat_curr_set')
features.remove('bat_volt_set')


# separate categorical and numerical features

categorical = ['sys_phase',
               'aux_wp',
               'trac_wp',
               'dcdc1_inv_stat',
               'dcdc2_inv_stat',
               'dcac3_inv_stat']

numerical = ['gfd_res',
             'bat_soc',
             'aux_cw_temp',
             'aux_hw_temp',
             'trac_cw_temp',
             'trac_hw_temp',
             'aircomp_speed',
             'aircomp_temp',
             'dcdc1_power',
             'dcdc1_temp',
             'dcdc2_power',
             'dcdc2_temp',
             'dcac3_power',
             'dcac3_temp']


for var in numerical:
    
    print(var, ':', eval(var).value.nunique())
    

for var in categorical:
    
    print(var, ':', eval(var).value.nunique())


# round timestamps and create time index

for var in features:
    
    eval(var).time = eval(var).time.dt.round('s')


start_time = min([eval(x).time.min() for x in features])
end_time = max([eval(x).time.max() for x in features])

time_index = pd.date_range(start=start_time, end=end_time, freq='s')


# create master df_raw

df_raw = pd.DataFrame(time_index, columns=['time'])


# merge features into master df_raw

for var in features:
    
    df_raw = pd.merge(df_raw, eval(var).rename(columns={'value': var}), on='time', how='left')
        
df_raw['time'] = df_raw['time'].dt.tz_localize(None)  

df_raw.set_index('time', inplace=True)  


# convert water pump values to binary

df_raw['aux_wp'] = df_raw['aux_wp'].replace(3150,1)
df_raw['trac_wp'] = df_raw['aux_wp'].replace(2600,1)


# adjust sys_phase to provided description

df_raw['sys_phase'] = df_raw['sys_phase'].add(1)







# Plot Time STEPS

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Numerical

df_raw[numerical].count().sort_values(ascending=False).plot(kind='bar', ax=ax1, color='royalblue', edgecolor='black')

ax1.set_title('Numerical Features')
#ax1.set_xlabel('Features')
ax1.set_ylabel('Count of Time Steps')

# Categorical

df_raw[categorical].count().sort_values(ascending=False).plot(kind='bar', ax=ax2,color='salmon', edgecolor='black')
ax2.set_title('Categorical Features')
#ax2.set_xlabel('Features')
ax2.set_ylabel('Count of Time Steps')

# Rotate tick labels

for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=80)
    
# Define subplot labels and axes for a loop

axes = [ax1, ax2]
labels = ['A', 'B']

for i, ax in enumerate(axes):

    ax.text(-0.1, 1.1, labels[i], transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', ha='right')    
    
plt.tight_layout()
#plt.savefig('raw_time_steps.png')   
plt.show()






#################################

#### DOWNSAMPLING to minutes ####


#################################


agg_logic = {col: 'mean' for col in numerical}
agg_logic.update({col: lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan for col in categorical})

df_resampled = df_raw.resample('1min').agg(agg_logic)


'''

# Downsample to 1 minute, but only calculate the mean/mode if we have at least 30 valid seconds.

# Calculate the valid sample count per minute

sample_counts = df_raw.resample('1min').count()

# Set values to NaN where fewer than 30 seconds were recorded (50% completeness)

MIN_SAMPLES = 30

df_resampled = df_resampled.where(sample_counts >= MIN_SAMPLES, np.nan)

'''

# save to parquet

df_resampled.to_parquet('bus478.parquet')


