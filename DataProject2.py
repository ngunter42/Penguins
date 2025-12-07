import csv
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

studyName = []
sampleNumber = []
species = []
region = []
island = []
stage = []
individualID = []
clutchCompletion = []
dateEgg = []
culmenLength = []   # mm
culmenDepth = []    # mm
flipperLength = []  # mm
bodyMass = []       # g
sex = []
delta15N = []
delta13C = []
comments = []

#Read CSV and remove missing data

with open('penguins.txt', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip header

    for row in reader:
        # remove any row with NA in important numeric fields
        if (
            row[9] == "NA" or row[10] == "NA" or row[11] == "NA" or
            row[12] == "NA" or row[13] == "NA"
        ):
            continue

        studyName.append(row[0])
        sampleNumber.append(row[1])
        species.append(row[2])
        region.append(row[3])
        island.append(row[4])
        stage.append(row[5])
        individualID.append(row[6])
        clutchCompletion.append(row[7])
        dateEgg.append(row[8])
        culmenLength.append(float(row[9]))
        culmenDepth.append(float(row[10]))
        flipperLength.append(float(row[11]))
        bodyMass.append(float(row[12]))
        sex.append(row[13].strip())
        delta15N.append(row[14])
        delta13C.append(row[15])
        comments.append(row[16])

# Remove outliers
def remove_outliers_by_group(species_list, sex_list, data_list):
    cleaned = data_list[:]

    groups = set(zip(species_list, sex_list))

    for sp, sx in groups:
        idx = [i for i in range(len(data_list)) if species_list[i] == sp and sex_list[i] == sx]
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
                cleaned[i] = None  # mark for deletion

    return cleaned


culmenLength = remove_outliers_by_group(species, sex, culmenLength)
culmenDepth = remove_outliers_by_group(species, sex, culmenDepth)
flipperLength = remove_outliers_by_group(species, sex, flipperLength)
bodyMass = remove_outliers_by_group(species, sex, bodyMass)

# Get rid of rows marked for removal
final_indices = [i for i in range(len(culmenLength)) if
                 culmenLength[i] is not None and
                 culmenDepth[i] is not None and
                 flipperLength[i] is not None and
                 bodyMass[i] is not None]

# Rebuild cleaned lists
studyName       = [studyName[i] for i in final_indices]
sampleNumber    = [sampleNumber[i] for i in final_indices]
species         = [species[i] for i in final_indices]
region          = [region[i] for i in final_indices]
island          = [island[i] for i in final_indices]
stage           = [stage[i] for i in final_indices]
individualID    = [individualID[i] for i in final_indices]
clutchCompletion= [clutchCompletion[i] for i in final_indices]
dateEgg         = [dateEgg[i] for i in final_indices]
culmenLength    = [culmenLength[i] for i in final_indices]
culmenDepth     = [culmenDepth[i] for i in final_indices]
flipperLength   = [flipperLength[i] for i in final_indices]
bodyMass        = [bodyMass[i] for i in final_indices]
sex             = [sex[i] for i in final_indices]
delta15N        = [delta15N[i] for i in final_indices]
delta13C        = [delta13C[i] for i in final_indices]
comments        = [comments[i] for i in final_indices]


# save cleaned data to csv
penguins_cleaned = pd.DataFrame({
    "studyName": studyName,
    "sampleNumber": sampleNumber,
    "species": species,
    "region": region,
    "island": island,
    "stage": stage,
    "individualID": individualID,
    "clutchCompletion": clutchCompletion,
    "dateEgg": dateEgg,
    "culmenLength": culmenLength,
    "culmenDepth": culmenDepth,
    "flipperLength": flipperLength,
    "bodyMass": bodyMass,
    "sex": sex,
    "delta15N": delta15N,
    "delta13C": delta13C,
    "comments": comments
})

# Save to CSV
penguins_cleaned.to_csv("penguins_cleaned.csv", index=False)


#Sort data into species and sex for exploratory figures
unique_species = sorted(set(species))
unique_sex = sorted(set(sex))

for sp in unique_species:
    for sx in unique_sex:
        idx = [i for i in range(len(species)) if species[i] == sp and sex[i] == sx]

        if len(idx) == 0:
            continue



# Exploratory figures
plt.figure(figsize=(10, 6))
plt.hist(culmenLength, bins=20, alpha=0.6)
plt.title("Distribution of Culmen Length")
plt.xlabel("Culmen Length (mm)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(bodyMass, bins=20, alpha=0.6)
plt.title("Distribution of Body Mass")
plt.xlabel("Body Mass (g)")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(culmenLength, bodyMass, alpha=0.7)
plt.title("Culmen Length vs Body Mass")
plt.xlabel("Culmen Length (mm)")
plt.ylabel("Body Mass (g)")
plt.savefig("CulmenLength_BodyMass.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(flipperLength, bodyMass, alpha=0.7)
plt.title("Flipper Length vs Body Mass")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.savefig("FlipperLength_BodyMass.png")
plt.show()

#Species boxplots
def boxplot_by_species(values, title, ylabel):
    plt.figure(figsize=(9, 6))
    grouped = []

    for sp in unique_species:
        sp_values = [values[i] for i in range(len(values)) if species[i] == sp]
        grouped.append(sp_values)

    plt.boxplot(grouped, labels=unique_species)
    plt.title(title)
    plt.ylabel(ylabel)

    # dynamic filename
    filename = title.replace(" ", "_") + ".png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print("Saved:", filename)

    plt.show()
# Sex boxplots
def boxplot_by_sex(values, title, ylabel):
    plt.figure(figsize=(9, 6))
    grouped = []

    for sx in unique_sex:
        sx_values = [values[i] for i in range(len(values)) if sex[i] == sx]
        grouped.append(sx_values)

    plt.boxplot(grouped, labels=unique_sex)
    plt.title(title)
    plt.ylabel(ylabel)

    # dynamic filename
    filename = title.replace(" ", "_") + ".png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print("Saved:", filename)

    plt.show()

boxplot_by_species(culmenLength, "Culmen Length by Species", "Culmen Length (mm)")
boxplot_by_species(culmenDepth, "Culmen Depth by Species", "Culmen Depth (mm)")
boxplot_by_sex(bodyMass,"Body Mass by Sex", "Body Mass (g)")
boxplot_by_sex(flipperLength, "Flipper Length by Sex", "Flipper Length (mm)")


# Regression
# Convert sex to 0 or 1
sex_numeric = []
for s in sex:
    if s == "FEMALE":
        sex_numeric.append(0)
    elif s == "MALE":
        sex_numeric.append(1)

X = np.column_stack([culmenLength, flipperLength, sex_numeric])
X = sm.add_constant(X)


y = np.array(bodyMass)


model = sm.OLS(y, X).fit()

print("\n===== REGRESSION MODEL SUMMARY =====")
print(model.summary())


pred = model.predict(X)
residuals = y - pred

# Residual and predicted vs actual plots
plt.figure(figsize=(8, 6))
plt.scatter(pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
title = "Residuals vs Predicted Body Mass"
plt.title(title)
plt.xlabel("Predicted Body Mass (g)")
plt.ylabel("Residuals")

filename = title.replace(" ", "_") + ".png"
plt.tight_layout()
plt.savefig(filename, dpi=300)
print("Saved:", filename)

plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y, pred, alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
title = "Predicted vs Actual Body Mass"
plt.title(title)
plt.xlabel("Actual Body Mass (g)")
plt.ylabel("Predicted Body Mass (g)")

filename = title.replace(" ", "_") + ".png"
plt.tight_layout()
plt.savefig(filename, dpi=300)
print("Saved:", filename)

plt.show()