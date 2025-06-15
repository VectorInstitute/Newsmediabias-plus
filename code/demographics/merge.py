import pandas as pd

"""
This script merges gender and topics demographic predictions with the main demo_analysis file.
"""

# Load CSVs
demo_analysis = pd.read_csv("./demo_analysis.csv")
gender = pd.read_csv("./gender.csv")
topics = pd.read_csv("./topics.csv")

# Merge on 'unique_id'
merged = demo_analysis.merge(gender, on='unique_id', how='left')
merged = merged.merge(topics, on='unique_id', how='left')

# Save updated file
merged.to_csv("./demo_analysis.csv", index=False)

# Print info
print("Merged dataset info:")
print(merged.info())
print("\nGender info:")
print(gender.info())
print("\nTopics info:")
print(topics.info())
