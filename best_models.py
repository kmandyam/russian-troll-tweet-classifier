from simpletransformers.classification import ClassificationModel
import pandas as pd

train_df = pd.read_csv("data/data_splits/train.csv")
val_df = pd.read_csv("data/data_splits/validation.csv")
eval_df = pd.read_csv("data/data_splits/test.csv")

dfs = [train_df, val_df, eval_df]

for df in dfs:
    df['tweet'] = df['tweet'].astype('str')
    df['label'] = df['label'].astype('int')

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base')

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)