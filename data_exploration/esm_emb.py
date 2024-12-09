from transformers import AutoTokenizer, EsmModel, pipeline
import torch
import pandas as pd
import tqdm

# Load the model and tokenizer
model_path = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = EsmModel.from_pretrained(model_path)

# Create a pipeline
esm_pipeline = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Read the protein_data DataFrame from a CSV file
df = pd.read_csv("./data_exploration/data/ProtSpatial/protein_data raw.csv", index_col=0)
df["Plate"] = df["Plate"].str.replace("Plate_", "").astype(int)

import pickle
# Process each sequence using the pipeline
i = 0
out = esm_pipeline(df["variant"].tolist()[:20])

# Save the output to a pickle file
with open("./data_exploration/data/ProtSpatial/esm_features.pkl", "wb") as f:
    pickle.dump(out, f)