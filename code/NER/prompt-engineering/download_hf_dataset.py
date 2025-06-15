from datasets import load_dataset
import pandas as pd

# Load the dataset in streaming mode
streamed_dataset = load_dataset("vector-institute/newsmediabias-plus")

ids = ['654ef428fe', '1b9b6f509f', 'c9b2579d06', '5bf0ec3a3d', 'e3ce687bf4', 
       'db6f13a554', '6f98a62554', 'd173f68814', '4d35e03150', '2b83650244']

# Filter the dataset

df = streamed_dataset['train'].to_pandas()
filtered_dataset = df[df["unique_id"].isin(ids)]

print(f"found {len(filtered_dataset)} rows")
# Collect the first 10 rows
first_10_rows = []
for i, row in enumerate(filtered_dataset['train']):
    try:
        # Process the data as needed, e.g., extract text, code, or other features
        text = row['article_text']
        first_10_rows.append(row)
        # ... further processing
    except Exception as e:
        print(f"Error processing row {i}: {e}")
    

# Convert the collected rows to a Pandas DataFrame
df = pd.DataFrame(first_10_rows)

# Write the DataFrame to a CSV file
df.to_csv('news_articles.csv', index=True, columns=['unique_id', 'article_text', 'text_label'])
# df.to_csv('newsmediabias_plus_10rows.csv', index=False)