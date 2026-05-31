import pandas as pd

df = pd.read_csv("data/Resume/Resume.csv")
print(df.shape)          # should be ~(2484, 4)
print(df.columns.tolist())  # should be ['ID', 'Resume_str', 'Resume_html', 'Category']
print(df["Category"].unique())  # should show clean labels like 'Data Science', 'HR', etc.