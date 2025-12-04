# %%

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

# %%
bpi_challenge_2013 = Path(
    r'C:\Users\Jonas\Desktop\real_logs\paper\results\bpi_challenge_2013_without_lifecycle_VARIANTS_NEW.csv')

bpi_challenge_2012_2_cases = Path(
    r'C:\Users\Jonas\Desktop\real_logs\paper\results\bpi_challenge_2012_2_cases_01_noise_VARIANTS_NEW.csv')

bpi_challenge_2012_3_cases = Path(
    r'C:\Users\Jonas\Desktop\real_logs\paper\results\bpi_challenge_2012_3_cases_01_noise_VARIANTS_NEW.csv')

bpi_challenge_2017_3_cases = Path(
    r'C:\Users\Jonas\Desktop\real_logs\paper\results\bpi_challenge_2017_3_cases_01_noise_VARIANTS_NEW.csv')

road_traffic_fines = Path(
    r'C:\Users\Jonas\Desktop\real_logs\paper\results\road_traffic_01_noise_VARIANTS_NEW.csv')

environmental_permit = Path(
    r'C:\Users\Jonas\Desktop\real_logs\paper\results\environmental_permit_01_noise_VARIANTS_NEW.csv')

real_log_paths = [bpi_challenge_2013, bpi_challenge_2012_2_cases, bpi_challenge_2012_3_cases,
                  bpi_challenge_2017_3_cases, road_traffic_fines, environmental_permit]

xixi_bpi_challenge_2013 = Path(
    r'C:\Users\Jonas\Desktop\real_logs\paper\results\exp_xixi_real_logs_cd_1\result_new_bpi_2013_challenge_final.csv')

xixi_road_traffic_fines = Path(
    r'C:\Users\Jonas\Desktop\real_logs\paper\results\exp_xixi_real_logs_cd_1\result_new_road_traffic_01_noise_final.csv')

xixi_environmental_permit = Path(
    r'C:\Users\Jonas\Desktop\real_logs\paper\results\exp_xixi_real_logs_cd_1\result_env_permit_01_noise_fitness.csv')

xixi_real_logs_paths = [xixi_bpi_challenge_2013, xixi_road_traffic_fines, xixi_environmental_permit]
# xixi_real_logs_paths = [xixi_environmental_permit]

# %%

xixi_dfs_real_logs = []
for path in xixi_real_logs_paths:
    # if 'png' in path:
    #     continue
    temp = pd.read_csv(path)
    temp['refined_f1_score'] = 2 * (temp['Refined Log Precision'] * temp['Refined Log Fitness']) / (
            temp['Refined Log Precision'] + temp['Refined Log Fitness'])

    print('path')
    print(path)
    print('Runtime')
    print(temp['Runtime'].mean())
    # print(temp['Runtime'].max())

    print('Precision')
    print(temp['Refined Log Precision'].mean())
    print(temp['Refined Log Precision'].max())

    print('Refined Fitness')
    print(temp['Refined Log Fitness'].mean())
    print(temp['Refined Log Fitness'].max())

    print('F1-score')
    print(temp['refined_f1_score'].mean())
    print(temp['refined_f1_score'].max())

    xixi_dfs_real_logs.append(temp)

xixi_bpi_challenge_2013 = xixi_dfs_real_logs[0]
xixi_road_traffic_fines = xixi_dfs_real_logs[1]
xixi_environmental_permit = xixi_dfs_real_logs[2]

xixi_df_real_logs_combined = pd.DataFrame(np.concatenate([df_temp.values for df_temp in xixi_dfs_real_logs]),
                                     columns=xixi_dfs_real_logs[0].columns)
# %%

dfs_real_logs = []
for path in real_log_paths:
    # if 'png' in path:
    #     continue
    temp = pd.read_csv(path)
    temp['refined_f1_score'] = 2 * (temp['Precision Align'] * temp['Fitness']) / (
            temp['Precision Align'] + temp['Fitness'])
    temp['unrefined_f1_score'] = 2 * (temp['original_precision'] * temp['original_fitness']) / (
            temp['original_precision'] + temp['original_fitness'])

    print('path')
    print(path)
    print('Runtime')
    print(temp['Runtime'].mean())
    # print(temp['Runtime'].max())

    print('Precision')
    print(temp['Precision Align'].mean())
    print(temp['Precision Align'].max())
    print('Original Precision')
    print(temp['original_precision'].mean())
    print('Fitness')
    print(temp['Fitness'].mean())
    print(temp['Fitness'].max())
    print('Original Fitness')
    print(temp['original_fitness'].mean())

    print('F1-score')
    print(temp['refined_f1_score'].mean())
    print(temp['refined_f1_score'].max())
    print('unrefined_f1_score')
    print(temp['unrefined_f1_score'].mean())

    dfs_real_logs.append(temp)

df_bpi_challenge_2013 = dfs_real_logs[0]
df_bpi_challenge_2012_2_cases = dfs_real_logs[1]
df_bpi_challenge_2012_3_cases = dfs_real_logs[2]
df_bpi_challenge_2017_3_cases = dfs_real_logs[3]
df_road_traffic_fines = dfs_real_logs[4]
df_environmental_permit = dfs_real_logs[5]

df_real_logs_combined = pd.DataFrame(np.concatenate([df_temp.values for df_temp in dfs_real_logs]),
                                     columns=dfs_real_logs[0].columns)

# %%

plt.figure()
fig, ax = plt.subplots()
bp1 = ax.boxplot(df_bpi_challenge_2012_3_cases['Precision Align'], positions=[1], patch_artist=True, boxprops=dict(facecolor="C0"))
bp2 = ax.boxplot(df_bpi_challenge_2013['Precision Align'], positions=[2], patch_artist=True, boxprops=dict(facecolor="C2"))
bp3 = ax.boxplot(xixi_bpi_challenge_2013['Refined Log Precision'], positions=[3], patch_artist=True, boxprops=dict(facecolor="C2"))
bp4 = ax.boxplot(df_bpi_challenge_2017_3_cases['Precision Align'], positions=[4], patch_artist=True, boxprops=dict(facecolor="C6"))
bp5 = ax.boxplot(df_road_traffic_fines['Precision Align'], positions=[5], patch_artist=True, boxprops=dict(facecolor="C7"))
bp8 = ax.boxplot(xixi_road_traffic_fines['Refined Log Precision'], positions=[6], patch_artist=True, boxprops=dict(facecolor="C7"))
bp6 = ax.boxplot(df_environmental_permit['Precision Align'], positions=[7], patch_artist=True, boxprops=dict(facecolor="C8"))
bp7 = ax.boxplot(xixi_environmental_permit['Refined Log Precision'], positions=[8], patch_artist=True, boxprops=dict(facecolor="C8"))


line_1 =ax.hlines(y=0.31, xmin=0.85, xmax=1.15, color='r') # BPI 2012 Precision
ax.hlines(y=0.8, xmin=1.85, xmax=2.15, color='r') # BPI 2013
ax.hlines(y=0.8, xmin=2.85, xmax=3.15, color='r') # BPI 2013 xixi
ax.hlines(y=0.37, xmin=3.85, xmax=4.15, color='r') # BPI 2017
ax.hlines(y=0.56, xmin=4.85, xmax=5.15, color='r') # Road Traffic Fines
ax.hlines(y=0.56, xmin=5.85, xmax=6.15, color='r') # Road Traffic Fines Xixi
ax.hlines(y=0.19, xmin=6.85, xmax=7.15, color='r') # Environmental Permit
ax.hlines(y=0.19, xmin=7.85, xmax=8.15, color='r') # Environmental Permit Xixi

ax.legend([bp1["boxes"][0], bp2["boxes"][0],
           bp4["boxes"][0], bp5["boxes"][0], bp6["boxes"][0], line_1],
          ['BPI Challenge 2012', 'BPI Challenge 2013',
           'BPI Challenge 2017', 'Road Traffic Fines', 'Environmental Permit', 'Unrefined Log'],
          loc='upper center', bbox_to_anchor=(0.5, 1.16),
          ncol=3, fancybox=True
          )
plt.xticks(list(range(1, 9)), ['Context-\nbased', 'Context-\nbased', 'Case\nMapping',
                               'Context-\nbased', 'Context-\nbased', 'Case\nMapping', 'Context-\nbased', 'Case\nMapping'])
plt.subplots_adjust(bottom=0.15)

plt.xlabel('Algorithm')
plt.ylabel('Precision')
plt.savefig(r'C:\Users\Jonas\Desktop\real_logs\plot_pngs\average_precision_real_logs.png', dpi=300)
plt.show()

# %%

plt.figure()
fig, ax = plt.subplots()
bp3 = ax.boxplot(xixi_bpi_challenge_2013['Refined Log Precision'], positions=[1], patch_artist=True, boxprops=dict(facecolor="C0"))
bp8 = ax.boxplot(xixi_road_traffic_fines['Refined Log Precision'], positions=[2], patch_artist=True, boxprops=dict(facecolor="C0"))
bp7 = ax.boxplot(xixi_environmental_permit['Refined Log Precision'], positions=[3], patch_artist=True, boxprops=dict(facecolor="C0"))


ax.hlines(y=0.8, xmin=0.85, xmax=1.15, color='r') # BPI 2013 xixi
ax.hlines(y=0.56, xmin=1.85, xmax=2.15, color='r') # Road Traffic Fines Xixi
ax.hlines(y=0.19, xmin=2.85, xmax=3.15, color='r') # Environmental Permit Xixi
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

ax.legend([line_1],
          ['Unrefined Event Log Precision'],
          loc='upper center', bbox_to_anchor=(0.5, 1.16),
          ncol=1, fancybox=True
          )
plt.xticks(list(range(1, 4)), ['BPI Challenge\n2013', 'Road Traffic\nFines', 'Environmental\nPermit'])
plt.subplots_adjust(bottom=0.15)

plt.xlabel('Event Log')
plt.ylabel('Precision')
plt.savefig(r'C:\Users\Jonas\Desktop\real_logs\plot_pngs\average_precision_real_logs_xixi.png', dpi=300)
plt.show()


# %%

plt.figure()
fig, ax = plt.subplots()
bp1 = ax.boxplot(df_bpi_challenge_2012_3_cases['Precision Align'], positions=[1], patch_artist=True, boxprops=dict(facecolor="C0"))
bp2 = ax.boxplot(df_bpi_challenge_2013['Precision Align'], positions=[2], patch_artist=True, boxprops=dict(facecolor="C0"))
bp4 = ax.boxplot(df_bpi_challenge_2017_3_cases['Precision Align'], positions=[3], patch_artist=True, boxprops=dict(facecolor="C0"))
bp5 = ax.boxplot(df_road_traffic_fines['Precision Align'], positions=[4], patch_artist=True, boxprops=dict(facecolor="C0"))
bp6 = ax.boxplot(df_environmental_permit['Precision Align'], positions=[5], patch_artist=True, boxprops=dict(facecolor="C0"))

ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax.set_ylim([0, 1])

line_1 =ax.hlines(y=0.31, xmin=0.85, xmax=1.15, color='r') # BPI 2012 Precision
ax.hlines(y=0.8, xmin=1.85, xmax=2.15, color='r') # BPI 2013
ax.hlines(y=0.37, xmin=2.85, xmax=3.15, color='r') # BPI 2017
ax.hlines(y=0.56, xmin=3.85, xmax=4.15, color='r') # Road Traffic Fines
ax.hlines(y=0.19, xmin=4.85, xmax=5.15, color='r') # Environmental Permit

ax.legend([line_1],
          ['Unrefined Event Log Precision'],
          loc='upper center', bbox_to_anchor=(0.5, 1.16),
          ncol=3, fancybox=True
          )
plt.xticks(list(range(1, 6)), ['BPI Chall\n2012', 'BPI Chall\n2013', 'BPI Chall\n2017', 'Road Traffic\nFines', 'Environmental\nPermit'])
plt.subplots_adjust(bottom=0.15)

plt.xlabel('Event Log')
plt.ylabel('Precision')
plt.savefig(r'C:\Users\Jonas\Desktop\real_logs\plot_pngs\average_precision_real_logs_paper.png', dpi=300)
plt.show()
