# Cuttof based on distribution of the data
# +- median is outlier

# Features: 
# 1. Well info
# 2. neighbor fitness
# 3. x y
# 4. reads
# 5. WT-close
# 6. bio python: ratio T/G
# 7. bio python: ratio A/C
# 8. Distance from positive control
# 9. Just Border 
# 10. 3 stage border in middle out
# 11. Counts per well post-hoc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("./data_exploration/data/ProtSpatial/protein_data raw.csv", index_col=0)

df["Plate"] = df["Plate"].str.replace("Plate_", "").astype(int)
print(df.head())
#unique values in each column
print(df.nunique())

metric = "Fitness_raw"

# Assuming df["variant"] is a list or Series of sequences
sequences = df["variant"]

# Find positions where at least one sequence differs
different_positions = set()
for i in range(len(sequences[0])):  # Loop over character positions
    column_chars = {seq[i] for seq in sequences}  # Collect characters at this position
    if len(column_chars) > 1:  # Check if there's a difference
        different_positions.add(i)

set_mutants_pos = sorted(different_positions)

df["mutant"] = df["variant"].apply(lambda x: "".join([x[i] for i in set_mutants_pos]))
WT = df["mutant"].value_counts().index[0]

#convert well to row and column eg E4 -> 5, 4

def well_to_row_col(well):
    well = well.replace(' ', '')
    row = ord(well[0]) - ord('A')
    col = int(well[1:]) - 1
    return row, col

df['row'], df['col'] = zip(*df['Position'].map(well_to_row_col))

#add little bit of noise to row and col
# df['row'] += np.random.normal(0, 0.1, size=len(df))
# df['col'] += np.random.normal(0, 0.1, size=len(df))


center_row = df['row'].unique().mean()
center_col = df['col'].unique().mean()
print(center_row, center_col)

df['crow'] = df['row'] - center_row
df['ccol'] = df['col'] - center_col

# df['dist'] = np.sqrt(df['crow']**2 + df['ccol']**2)
#manhattan distance
df['dist'] = np.abs(df['crow']) + np.abs(df['ccol'])

df['dist_chebychev'] = np.maximum(np.abs(df['crow']), np.abs(df['ccol']))

#dist_c_row
df['dist_c_row'] = np.abs(df['crow'])
#dist_c_col
df['dist_c_col'] = np.abs(df['ccol'])
df["border"] = (df["dist_c_row"] > 3 ) | ( df["dist_c_col"] > 5)
df["inner_square"] = (df["dist_chebychev"] < 2)
df["inner_rect"] = (df["dist_c_row"] < 2) & (df["dist_c_col"] < 3) 

# Define a function to get neighboring fitness within each plate
def get_neighbor_fitness(df, metric, direction):
    offsets = {
        "north": (1, 0),
        "south": (-1, 0),
        "west": (0, -1),
        "east": (0, 1)
    }
    row_offset, col_offset = offsets[direction]
    
    # Create temporary columns for shifted rows and columns
    temp_df = df.copy()
    temp_df['neighbor_row'] = temp_df['row'] + row_offset
    temp_df['neighbor_col'] = temp_df['col'] + col_offset

    # Merge with the original DataFrame to find the fitness of neighbors within the same plate
    neighbor_fitness = temp_df.merge(
        df[['Plate', 'row', 'col', metric]], 
        left_on=['Plate', 'neighbor_row', 'neighbor_col'], 
        right_on=['Plate', 'row', 'col'], 
        how='left', 
        suffixes=('', '_neighbor')
    )

    # Fill NaN with -1 for border wells
    neighbor_fitness[f'f_{direction}'] = neighbor_fitness[f'{metric}_neighbor'].fillna(-1)
    return neighbor_fitness[f'f_{direction}']

# Generate the four features for neighboring fitness, considering the Plate column
df['f_north'] = get_neighbor_fitness(df, metric, 'north')
df['f_south'] = get_neighbor_fitness(df, metric, 'south')
df['f_west'] = get_neighbor_fitness(df, metric, 'west')
df['f_east'] = get_neighbor_fitness(df, metric, 'east')

# Display the updated DataFrame
print(df[['Plate', 'row', 'col', 'f_north', 'f_south', 'f_west', 'f_east']].head())


# sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=df['std_norm_TSNAK'])
fig = plt.figure(figsize=(4, 4))

df_pl = df.select_dtypes(include=[np.number]).copy()
df_c = df_pl.groupby(['row', 'col']).mean().reset_index()
sns.scatterplot(x='col', y='row', hue=metric, data=df[df["Plate"] == 1], s=200, edgecolor='black', legend="full")
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#title
plt.title('Counts per well')
# bigger
plt.gcf().set_size_inches(6, 6)
# plt.show()

df["replicate_count"] = df.groupby(['variant'])[metric].transform('count')
df = df[df["replicate_count"] > 3]

df["median"] = df.groupby(['variant'])[metric].transform('median')
df["std"] = df.groupby(['variant'])[metric].transform('std')
 
# df = df[df["mutant"] == WT]

df["count_per_well"] = df_pl.groupby(['row', 'col']).transform('count')[metric] 
df["outlier"] = (df[metric] > df["median"] + df["std"]) | (df[metric] < df["median"] - df["std"])

outlier_df = df.select_dtypes(include=[np.number, bool]).reset_index(drop=True)

#drop fitness_raw & fitness
outlier_df = outlier_df.drop(columns=[metric, "Fitness", "replicate_count", "median", "std"])
print(outlier_df)
print(f"Outlier count: {outlier_df['outlier'].sum()}")

#write to csv
outlier_df.to_csv("./data_exploration/data/ProtSpatial/outlier_df.csv")