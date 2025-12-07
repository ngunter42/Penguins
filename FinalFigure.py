import csv
import numpy as np
import matplotlib.pyplot as plt

# Same CSV load in as first file but simplified and adjusted for final figures
studyName = []
sampleNumber = []
species = []
culmenLength = []   # mm
culmenDepth = []    # mm
sex = []

with open('penguins.txt', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip header

    for row in reader:
        if row[9] == "NA" or row[10] == "NA":
            continue

        studyName.append(row[0])
        sampleNumber.append(row[1])
        species.append(row[2].strip())
        culmenLength.append(float(row[9]))
        culmenDepth.append(float(row[10]))
        sex.append(row[13].strip())

# Standardize species name to common name
species_cleaned = []
for s in species:
    s_lower = s.lower().replace("é","e")
    if "adelie" in s_lower:
        species_cleaned.append("Adelie")
    elif "chinstrap" in s_lower:
        species_cleaned.append("Chinstrap")
    elif "gentoo" in s_lower:
        species_cleaned.append("Gentoo")
    else:
        species_cleaned.append(s)
species = species_cleaned
unique_species = ["Adelie", "Chinstrap", "Gentoo"]


def remove_outliers_by_group(species_list, data_list):
    cleaned = data_list[:]
    for sp in set(species_list):
        idx = [i for i in range(len(data_list)) if species_list[i] == sp]
        values = [data_list[i] for i in idx]
        if len(values) < 4:
            continue
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        for i in idx:
            if data_list[i] < lower or data_list[i] > upper:
                cleaned[i] = None
    return cleaned

culmenLength = remove_outliers_by_group(species, culmenLength)
culmenDepth = remove_outliers_by_group(species, culmenDepth)


final_indices = [i for i in range(len(culmenLength)) if culmenLength[i] is not None and culmenDepth[i] is not None]
species = [species[i] for i in final_indices]
culmenLength = [culmenLength[i] for i in final_indices]
culmenDepth = [culmenDepth[i] for i in final_indices]

# Color and font
species_colors = {"Adelie": "#0072B2", "Chinstrap": "#D55E00", "Gentoo": "#009E73"}
title_fs = 28
label_fs = 24
tick_fs = 28
n_fs = 24

# Add n= sample size
def add_sample_sizes(ax, groups):
    ylim = ax.get_ylim()
    y_pos = ylim[0] - 0.15 * (ylim[1]-ylim[0])
    for i, group in enumerate(groups):
        n = len(group)
        ax.text(i+1, y_pos, f"n={n}", ha='center', va='top', fontsize=n_fs)

# Culmen Length boxplot
culmen_length_groups = [[culmenLength[i] for i in range(len(species)) if species[i]==sp] for sp in unique_species]

fig, ax = plt.subplots(figsize=(12,8))
bplot = ax.boxplot(culmen_length_groups, labels=unique_species, patch_artist=True)
for patch, sp in zip(bplot['boxes'], unique_species):
    patch.set_facecolor(species_colors[sp])
for median in bplot['medians']:
    median.set(color='black', linewidth=2)

ax.set_title("Culmen Length by Species", fontsize=title_fs)
ax.set_ylabel("Culmen Length (mm)", fontsize=label_fs)
ax.tick_params(axis='x', labelsize=tick_fs)
ax.tick_params(axis='y', labelsize=tick_fs)

add_sample_sizes(ax, culmen_length_groups)
plt.tight_layout()
plt.savefig("CulmenLength_by_Species_final.png", dpi=300)
plt.show()

# Culmen depth boxplot
culmen_depth_groups = [[culmenDepth[i] for i in range(len(species)) if species[i]==sp] for sp in unique_species]

fig, ax = plt.subplots(figsize=(12,8))
bplot = ax.boxplot(culmen_depth_groups, labels=unique_species, patch_artist=True)
for patch, sp in zip(bplot['boxes'], unique_species):
    patch.set_facecolor(species_colors[sp])
for median in bplot['medians']:
    median.set(color='black', linewidth=2)

ax.set_title("Culmen Depth by Species", fontsize=title_fs)
ax.set_ylabel("Culmen Depth (mm)", fontsize=label_fs)
ax.tick_params(axis='x', labelsize=tick_fs)
ax.tick_params(axis='y', labelsize=tick_fs)

add_sample_sizes(ax, culmen_depth_groups)
plt.tight_layout()
plt.savefig("CulmenDepth_by_Species_final.png", dpi=300)
plt.show()
