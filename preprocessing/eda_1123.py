#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
# import matlib


# In[4]:


com = pd.read_csv('preprocessed_data/jiwon_combined_data.csv', index_col=0)
com


# In[5]:


import pandas as pd

df = com

# Group the data by 'year'
yearly_groups = df.groupby('year')

# Function to perform EDA for each year
def perform_yearly_eda(groups):
    eda_results = {}
    for year, group in groups:
        eda_results[year] = {
            "Row Count": len(group),
            "AGE2 Stats": group['AGE2'].describe(),
            "IRSEX Distribution": group['IRSEX'].value_counts(),
            "IRMARIT Distribution": group['IRMARIT'].value_counts(),
            "SERVICE Distribution": group['SERVICE'].value_counts(),
            "CG30EST Distribution": group['CG30EST'].value_counts(),
            "AL30EST Distribution": group['AL30EST'].value_counts(),
        }
    return eda_results

# Perform EDA
eda_results = perform_yearly_eda(yearly_groups)

# Print EDA results for each year
for year, stats in eda_results.items():
    print(f"\nYear: {year}")
    print(f"Row Count: {stats['Row Count']}")
    print(f"AGE2 Stats:\n{stats['AGE2 Stats']}")
    print(f"IRSEX Distribution:\n{stats['IRSEX Distribution']}")
    print(f"IRMARIT Distribution:\n{stats['IRMARIT Distribution']}")
    print(f"SERVICE Distribution:\n{stats['SERVICE Distribution']}")
    print(f"CG30EST Distribution:\n{stats['CG30EST Distribution']}")
    print(f"AL30EST Distribution:\n{stats['AL30EST Distribution']}")


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = com

# Group the data by 'year'
yearly_groups = df.groupby('year')

# Function to visualize yearly distributions
def visualize_yearly_eda(groups):
    for year, group in groups:
        print(f"Visualizing data for year: {year}")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        fig.suptitle(f"Yearly EDA for {year}", fontsize=16)

        # AGE2 - Histogram
        sns.histplot(group['AGE2'], bins=10, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title("AGE2 Distribution")
        axes[0, 0].set_xlabel("AGE2")
        axes[0, 0].set_ylabel("Frequency")

        # IRSEX - Bar Plot
        group['IRSEX'].value_counts(sort=False).sort_index().plot(kind='bar', ax=axes[0, 1], color='skyblue')
        axes[0, 1].set_title("IRSEX Distribution")
        axes[0, 1].set_xlabel("IRSEX")
        axes[0, 1].set_ylabel("Count")

        # IRMARIT - Bar Plot
        group['IRMARIT'].value_counts(sort=False).sort_index().plot(kind='bar', ax=axes[1, 0], color='lightgreen')
        axes[1, 0].set_title("IRMARIT Distribution")
        axes[1, 0].set_xlabel("IRMARIT")
        axes[1, 0].set_ylabel("Count")

        # SERVICE - Bar Plot
        group['SERVICE'].value_counts(sort=False).sort_index().plot(kind='bar', ax=axes[1, 1], color='salmon')
        axes[1, 1].set_title("SERVICE Distribution")
        axes[1, 1].set_xlabel("SERVICE")
        axes[1, 1].set_ylabel("Count")

        # CG30EST - Bar Plot
        group['CG30EST'].value_counts(sort=False).sort_index().plot(kind='bar', ax=axes[2, 0], color='orange')
        axes[2, 0].set_title("CG30EST Distribution")
        axes[2, 0].set_xlabel("CG30EST")
        axes[2, 0].set_ylabel("Count")

        # AL30EST - Bar Plot
        group['AL30EST'].value_counts(sort=False).sort_index().plot(kind='bar', ax=axes[2, 1], color='purple')
        axes[2, 1].set_title("AL30EST Distribution")
        axes[2, 1].set_xlabel("AL30EST")
        axes[2, 1].set_ylabel("Count")

        # Adjust layout and save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect for the title
        plt.savefig(f"eda_visualization_{year}.png")  # Save the visualization as a PNG
        plt.show()

# Call the visualization function
visualize_yearly_eda(yearly_groups)


# In[ ]:




