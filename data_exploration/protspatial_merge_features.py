import pandas as pd

# model_name = "esm2_t33_650M_UR50D"
model_name = "esm2_t6_8M_UR50D"
model_path = "facebook/" + model_name

# 2. Load and preprocess the dataset
df = pd.read_csv("./data_exploration/data/ProtSpatial/protein_data raw.csv", index_col=0)

csv_path = f"./data_exploration/data/ProtSpatial/{model_name}_embeddings.csv"
# merge data
df_emb = pd.read_csv(csv_path)
df.columns
# Index(['Fitness', 'Fitness_raw', 'Plate', 'Position', 'variant', 'Reads'], dtype='object')
df_emb.columns
# Index(['variant', 'embedding'], dtype='object')

#merge on variant
df_final = df.merge(df_emb, on='variant')
df_final.nunique()
df_final.to_csv(f"./data_exploration/data/ProtSpatial/protein_data{model_name}_final.csv")