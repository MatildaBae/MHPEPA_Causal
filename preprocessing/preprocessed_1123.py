#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

# File paths for each year
file_paths = {
#     "2002": "data/NSDUH_2002_Tab.tsv",  # Replace with actual file paths
#     "2003": "data/NSDUH_2003_Tab.tsv",
#     "2004": "data/NSDUH_2004_Tab.tsv",
    "2005": "data/NSDUH_2005_Tab.tsv",
    "2006": "data/NSDUH_2006_Tab.tsv",
    "2007": "data/NSDUH_2007_Tab.tsv",
    "2008": "data/NSDUH_2008_Tab.tsv",
    "2009": "data/NSDUH_2009_Tab.tsv",
    "2010": "data/NSDUH_2010_Tab.tsv",
    "2011": "data/NSDUH_2011_Tab.tsv",
    "2012": "data/NSDUH_2012_Tab.tsv",
    "2013": "data/NSDUH_2013_Tab.tsv",
    "2014": "data/NSDUH_2014_Tab.tsv",
    "2015": "data/NSDUH_2015_Tab.tsv",
    "2016": "data/NSDUH_2016_Tab.tsv",
    "2017": "data/NSDUH_2017_Tab.tsv",
    "2018": "data/NSDUH_2018_Tab.tsv",
    "2019": "data/NSDUH_2019_Tab.tsv",
    "2020": "data/NSDUH_2020_Tab.tsv"
}

# Columns to extract
columns = ['PRVHLTIN', 'GRPHLTIN', 'QUESTID2']

# Output folder
output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Process each file
for year, file_path in file_paths.items():
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        
        # Extract specified columns
        filtered_df = df[columns]
        
        # Save to new CSV file
        output_file = os.path.join(output_folder, f"treat_{year}.csv")
        filtered_df.to_csv(output_file)
        
        print(f"Processed {year}: Saved to {output_file}")
    except Exception as e:
        print(f"Error processing {year}: {e}")


# In[44]:


import pandas as pd
import os

# File paths for each year
file_paths = {
#     "2002": "data/NSDUH_2002_Tab.tsv",  # Replace with actual file paths
#     "2003": "data/NSDUH_2003_Tab.tsv",
#     "2004": "data/NSDUH_2004_Tab.tsv",
#     "2005": "data/NSDUH_2005_Tab.tsv",
#     "2006": "data/NSDUH_2006_Tab.tsv",
#     "2007": "data/NSDUH_2007_Tab.tsv",
#     "2008": "data/NSDUH_2008_Tab.tsv",
#     "2009": "data/NSDUH_2009_Tab.tsv",
#     "2010": "data/NSDUH_2010_Tab.tsv",
#     "2011": "data/NSDUH_2011_Tab.tsv",
#     "2012": "data/NSDUH_2012_Tab.tsv",
#     "2013": "data/NSDUH_2013_Tab.tsv",
#     "2014": "data/NSDUH_2014_Tab.tsv",
    "2015": "data/NSDUH_2015_Tab.tsv",
#     "2016": "data/NSDUH_2016_Tab.tsv",
#     "2017": "data/NSDUH_2017_Tab.tsv",
#     "2018": "data/NSDUH_2018_Tab.tsv",
#     "2019": "data/NSDUH_2019_Tab.tsv",
#     "2020": "data/NSDUH_2020_Tab.tsv"
}

# Columns to extract
columns = ['QUESTID2',
           'AGE2', 'IRSEX', 
           'IRMARITSTAT', 
           'SERVICE', 'CG30EST', 'AL30EST',
           'AUNMDOC2', 'AUNMCLN2', 'AUNMDTM2', 'AUNMOTO2',
           'IRHHSIZ2', 'IRKI17_2', 'NOBOOKY2',
           'INCOME', 'GOVTPROG', 'COCFLAG', 'COCYR', 
           'HEALTH', 'AUNMTHE2',
           'AUPOPAMT', 'AUUNCOST', 'AUUNNCOV', 'AUUNENUF',
          ]

# Output folder
output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Process each file
for year, file_path in file_paths.items():
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        
        # Extract specified columns
        filtered_df = df[columns]
        
        # Rename the column to 'SPDYR'
        filtered_df.rename(columns={'IRMARITSTAT': 'IRMARIT'}, inplace=True)
        
        # Save to new CSV file
        output_file = os.path.join(output_folder, f"everyone_{year}.csv")
        filtered_df.to_csv(output_file)
        
        print(f"Processed {year}: Saved to {output_file}")
    except Exception as e:
        print(f"Error processing {year}: {e}")


# In[45]:


import os
import pandas as pd

# Path to the folder containing the files
folder_path = "preprocessed_data"

# List to store individual DataFrames
dataframes = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv") and file_name.startswith("everyone_"):
        # Extract the year from the file name
        year = int(file_name.split("_")[1].split(".")[0])
        
        # Load the file into a DataFrame
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, index_col=0)
        
        # Add the 'year' column
        df['year'] = year
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df


# In[46]:


combined_df['year'].nunique()


# In[50]:


# Save the combined DataFrame to a new CSV file
output_file = "preprocessed_data/everyone_combined_data.csv"
combined_df.to_csv(output_file)


# ## SoHeeeee

# In[51]:


import pandas as pd
import os

# File paths for each year
file_paths = {
    "2002": "data/NSDUH_2002_Tab.tsv",  # Replace with actual file paths
    "2003": "data/NSDUH_2003_Tab.tsv",
    "2004": "data/NSDUH_2004_Tab.tsv",
    "2005": "data/NSDUH_2005_Tab.tsv",
    "2006": "data/NSDUH_2006_Tab.tsv",
    "2007": "data/NSDUH_2007_Tab.tsv",
    "2008": "data/NSDUH_2008_Tab.tsv",
    "2009": "data/NSDUH_2009_Tab.tsv",
    "2010": "data/NSDUH_2010_Tab.tsv",
    "2011": "data/NSDUH_2011_Tab.tsv",
    "2012": "data/NSDUH_2012_Tab.tsv",
    "2013": "data/NSDUH_2013_Tab.tsv",
    "2014": "data/NSDUH_2014_Tab.tsv",
#     "2015": "data/NSDUH_2015_Tab.tsv",
#     "2016": "data/NSDUH_2016_Tab.tsv",
#     "2017": "data/NSDUH_2017_Tab.tsv",
#     "2018": "data/NSDUH_2018_Tab.tsv",
#     "2019": "data/NSDUH_2019_Tab.tsv",
#     "2020": "data/NSDUH_2020_Tab.tsv"
}

# Columns to extract
columns = ['QUESTID2','EDUCCAT2', 'JBSTATR2','TXEVER', 'TXYREVER']

# Output folder
output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Process each file
for year, file_path in file_paths.items():
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        
        # Extract specified columns
        filtered_df = df[columns]
        
        # Save to new CSV file
        output_file = os.path.join(output_folder, f"sohee_4_{year}.csv")
        filtered_df.to_csv(output_file)
        
        print(f"Processed {year}: Saved to {output_file}")
    except Exception as e:
        print(f"Error processing {year}: {e}")


# In[52]:


import pandas as pd
import os

# File paths for each year
file_paths = {
#     "2002": "data/NSDUH_2002_Tab.tsv",  # Replace with actual file paths
#     "2003": "data/NSDUH_2003_Tab.tsv",
#     "2004": "data/NSDUH_2004_Tab.tsv",
#     "2005": "data/NSDUH_2005_Tab.tsv",
#     "2006": "data/NSDUH_2006_Tab.tsv",
#     "2007": "data/NSDUH_2007_Tab.tsv",
#     "2008": "data/NSDUH_2008_Tab.tsv",
#     "2009": "data/NSDUH_2009_Tab.tsv",
#     "2010": "data/NSDUH_2010_Tab.tsv",
#     "2011": "data/NSDUH_2011_Tab.tsv",
#     "2012": "data/NSDUH_2012_Tab.tsv",
#     "2013": "data/NSDUH_2013_Tab.tsv",
#     "2014": "data/NSDUH_2014_Tab.tsv",
    "2015": "data/NSDUH_2015_Tab.tsv",
    "2016": "data/NSDUH_2016_Tab.tsv",
    "2017": "data/NSDUH_2017_Tab.tsv",
    "2018": "data/NSDUH_2018_Tab.tsv",
    "2019": "data/NSDUH_2019_Tab.tsv",
    "2020": "data/NSDUH_2020_Tab.tsv"
}

# Columns to extract
columns = ['QUESTID2','IREDUHIGHST2', 'WRKSTATWK2', 'TXEVRRCVD', 'TXYRRECVD']

# Output folder
output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Process each file
for year, file_path in file_paths.items():
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        
        # Extract specified columns
        filtered_df = df[columns]
        
        # Rename the column to 'SPDYR'
        filtered_df.rename(columns={'IRMARITSTAT': 'IRMARIT',
                                    'IREDUHIGHST2' : 'EDUCCAT2',
                                    'WRKSTATWK2' : 'JBSTATR2',
                                    'TXEVRRCVD' : 'TXEVER',
                                    'TXYRRECVD' : 'TXYREVER'}, inplace=True)
        
        # Save to new CSV file
        output_file = os.path.join(output_folder, f"sohee_4_{year}.csv")
        filtered_df.to_csv(output_file)
        
        print(f"Processed {year}: Saved to {output_file}")
    except Exception as e:
        print(f"Error processing {year}: {e}")


# In[53]:


import os
import pandas as pd

# Path to the folder containing the files
folder_path = "preprocessed_data"

# List to store individual DataFrames
dataframes = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv") and file_name.startswith("sohee_4_"):
        # Extract the year from the file name
        year = int(file_name.split("_")[2].split(".")[0])
        
        # Load the file into a DataFrame
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, index_col=0)
        
        # Add the 'year' column
        df['year'] = year
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df


# In[58]:


combined_df['year'].nunique()


# In[59]:


# Save the combined DataFrame to a new CSV file
output_file = "preprocessed_data/sohee_4_combined_data.csv"
combined_df.to_csv(output_file)


# ## Jongrak

# In[15]:


import pandas as pd
import os

# File paths for each year
file_paths = {
    "2002": "data/NSDUH_2002_Tab.tsv",  # Replace with actual file paths
    "2003": "data/NSDUH_2003_Tab.tsv",
    "2004": "data/NSDUH_2004_Tab.tsv",
    "2005": "data/NSDUH_2005_Tab.tsv",
    "2006": "data/NSDUH_2006_Tab.tsv",
    "2007": "data/NSDUH_2007_Tab.tsv",
    "2008": "data/NSDUH_2008_Tab.tsv",
    "2009": "data/NSDUH_2009_Tab.tsv",
    "2010": "data/NSDUH_2010_Tab.tsv",
    "2011": "data/NSDUH_2011_Tab.tsv",
    "2012": "data/NSDUH_2012_Tab.tsv",
    "2013": "data/NSDUH_2013_Tab.tsv",
    "2014": "data/NSDUH_2014_Tab.tsv",
    "2015": "data/NSDUH_2015_Tab.tsv",
    "2016": "data/NSDUH_2016_Tab.tsv",
    "2017": "data/NSDUH_2017_Tab.tsv",
    "2018": "data/NSDUH_2018_Tab.tsv",
    "2019": "data/NSDUH_2019_Tab.tsv",
    "2020": "data/NSDUH_2020_Tab.tsv"
}

# Columns to extract
columns = ['AUPOPAMT', 'AUUNCOST', 'AUUNNCOV', 'AUUNENUF']

# Output folder
output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Process each file
for year, file_path in file_paths.items():
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        
        # Extract specified columns
        filtered_df = df[columns]
        
        # Save to new CSV file
        output_file = os.path.join(output_folder, f"jongrak_{year}.csv")
        filtered_df.to_csv(output_file)
        
        print(f"Processed {year}: Saved to {output_file}")
    except Exception as e:
        print(f"Error processing {year}: {e}")


# In[17]:


import os
import pandas as pd

# Path to the folder containing the files
folder_path = "preprocessed_data"

# List to store individual DataFrames
dataframes = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv") and file_name.startswith("jongrak_"):
        # Extract the year from the file name
        year = int(file_name.split("_")[1].split(".")[0])
        
        # Load the file into a DataFrame
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, index_col=0)
        
        # Add the 'year' column
        df['year'] = year
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df


# In[56]:


combined_df = combined_df[~combined_df['year'].isin([2000, 2001, 2002, 2003, 2004])]
combined_df


# In[57]:


combined_df['year'].nunique()


# In[20]:


# Save the combined DataFrame to a new CSV file
output_file = "preprocessed_data/jongrak_combined_data.csv"
combined_df.to_csv(output_file)


# ## Jiwon

# In[22]:


df = pd.read_csv('preprocessed_data/jiwon_combined_data.csv', index_col=0)
df


# In[27]:


df = df[~df['year'].isin([2000, 2001])]
df


# In[28]:


df['year'].nunique()


# In[29]:


df.to_csv('preprocessed_data/jiwon_combined_data.csv')


# ## Jeongje

# In[30]:


import pandas as pd
import os

# File paths for each year
file_paths = {
#     "2002": "data/NSDUH_2002_Tab.tsv",  # Replace with actual file paths
#     "2003": "data/NSDUH_2003_Tab.tsv",
#     "2004": "data/NSDUH_2004_Tab.tsv",
#     "2005": "data/NSDUH_2005_Tab.tsv",
#     "2006": "data/NSDUH_2006_Tab.tsv",
#     "2007": "data/NSDUH_2007_Tab.tsv",
    "2008": "data/NSDUH_2008_Tab.tsv",
    "2009": "data/NSDUH_2009_Tab.tsv",
    "2010": "data/NSDUH_2010_Tab.tsv",
    "2011": "data/NSDUH_2011_Tab.tsv",
    "2012": "data/NSDUH_2012_Tab.tsv",
    "2013": "data/NSDUH_2013_Tab.tsv",
    "2014": "data/NSDUH_2014_Tab.tsv",
    "2015": "data/NSDUH_2015_Tab.tsv",
    "2016": "data/NSDUH_2016_Tab.tsv",
    "2017": "data/NSDUH_2017_Tab.tsv",
    "2018": "data/NSDUH_2018_Tab.tsv",
    "2019": "data/NSDUH_2019_Tab.tsv",
    "2020": "data/NSDUH_2020_Tab.tsv"
}

# Columns to extract
columns = ['SPDYR']

# Output folder
output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Process each file
for year, file_path in file_paths.items():
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        
        # Extract specified columns
        filtered_df = df[columns]
        
        # Save to new CSV file
        output_file = os.path.join(output_folder, f"spd_{year}.csv")
        filtered_df.to_csv(output_file)
        
        print(f"Processed {year}: Saved to {output_file}")
    except Exception as e:
        print(f"Error processing {year}: {e}")


# In[31]:


import pandas as pd
import os

# File paths for each year
file_paths = {
#     "2002": "data/NSDUH_2002_Tab.tsv",  # Replace with actual file paths
#     "2003": "data/NSDUH_2003_Tab.tsv",
#     "2004": "data/NSDUH_2004_Tab.tsv",
    "2005": "data/NSDUH_2005_Tab.tsv",
    "2006": "data/NSDUH_2006_Tab.tsv",
    "2007": "data/NSDUH_2007_Tab.tsv",
#     "2008": "data/NSDUH_2008_Tab.tsv",
#     "2009": "data/NSDUH_2009_Tab.tsv",
#     "2010": "data/NSDUH_2010_Tab.tsv",
#     "2011": "data/NSDUH_2011_Tab.tsv",
#     "2012": "data/NSDUH_2012_Tab.tsv",
#     "2013": "data/NSDUH_2013_Tab.tsv",
#     "2014": "data/NSDUH_2014_Tab.tsv",
#     "2015": "data/NSDUH_2015_Tab.tsv",
#     "2016": "data/NSDUH_2016_Tab.tsv",
#     "2017": "data/NSDUH_2017_Tab.tsv",
#     "2018": "data/NSDUH_2018_Tab.tsv",
#     "2019": "data/NSDUH_2019_Tab.tsv",
#     "2020": "data/NSDUH_2020_Tab.tsv"
}

# Columns to extract
columns = ['SPDYRADJ']

# Output folder
output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Process each file
for year, file_path in file_paths.items():
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        
        # Extract specified columns
        filtered_df = df[columns]
        
        # Rename the column to 'SPDYR'
        filtered_df.rename(columns={'SPDYRADJ': 'SPDYR'}, inplace=True)
        
        # Save to new CSV file
        output_file = os.path.join(output_folder, f"spd_{year}.csv")
        filtered_df.to_csv(output_file)
        
        print(f"Processed {year}: Saved to {output_file}")
    except Exception as e:
        print(f"Error processing {year}: {e}")


# In[32]:


import os
import pandas as pd

# Path to the folder containing the files
folder_path = "preprocessed_data"

# List to store individual DataFrames
dataframes = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv") and file_name.startswith("spd_"):
        # Extract the year from the file name
        year = int(file_name.split("_")[1].split(".")[0])
        
        # Load the file into a DataFrame
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, index_col=0)
        
        # Add the 'year' column
        df['year'] = year
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df


# In[33]:


combined_df['year'].nunique()


# In[35]:


import pandas as pd
import matplotlib.pyplot as plt

# SPDYR 값의 분포 계산
df = combined_df
value_counts = df['SPDYR'].value_counts(dropna=False).sort_index()

# NaN을 'Missing'으로 대체하여 시각화에 포함
value_counts.index = value_counts.index.fillna('Missing')

# 바그래프 그리기
plt.figure(figsize=(10, 6))
value_counts.plot(kind='bar', width=0.8)

# 그래프 제목 및 축 레이블 설정
plt.title('Distribution of SPDYR Values', fontsize=16)
plt.xlabel('SPDYR', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# 값 레이블 추가
for index, value in enumerate(value_counts):
    plt.text(index, value + max(value_counts) * 0.01, str(value), ha='center', fontsize=10)

# 그래프 표시
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## 변수보기

# In[64]:


df_2016 = pd.read_csv("data/NSDUH_2010_Tab.tsv", sep="\t")
df_2016['K6SCMON']


# In[65]:


import pandas as pd
import matplotlib.pyplot as plt

# SPDYR 값의 분포 계산
df = df_2010
value_counts = df['K6SCMON'].value_counts(dropna=False).sort_index()

# NaN을 'Missing'으로 대체하여 시각화에 포함
value_counts.index = value_counts.index.fillna('Missing')

# 바그래프 그리기
plt.figure(figsize=(10, 6))
value_counts.plot(kind='bar', width=0.8)

# 그래프 제목 및 축 레이블 설정
plt.title('Distribution of SPDYR Values', fontsize=16)
plt.xlabel('K6SCMON', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# 값 레이블 추가
for index, value in enumerate(value_counts):
    plt.text(index, value + max(value_counts) * 0.01, str(value), ha='center', fontsize=10)

# 그래프 표시
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[66]:


import pandas as pd
import os

# File paths for each year
file_paths = {
#     "2002": "data/NSDUH_2002_Tab.tsv",  # Replace with actual file paths
#     "2003": "data/NSDUH_2003_Tab.tsv",
#     "2004": "data/NSDUH_2004_Tab.tsv",
    "2005": "data/NSDUH_2005_Tab.tsv",
    "2006": "data/NSDUH_2006_Tab.tsv",
    "2007": "data/NSDUH_2007_Tab.tsv",
#     "2008": "data/NSDUH_2008_Tab.tsv",
#     "2009": "data/NSDUH_2009_Tab.tsv",
#     "2010": "data/NSDUH_2010_Tab.tsv",
#     "2011": "data/NSDUH_2011_Tab.tsv",
#     "2012": "data/NSDUH_2012_Tab.tsv",
#     "2013": "data/NSDUH_2013_Tab.tsv",
#     "2014": "data/NSDUH_2014_Tab.tsv",
#     "2015": "data/NSDUH_2015_Tab.tsv",
#     "2016": "data/NSDUH_2016_Tab.tsv",
#     "2017": "data/NSDUH_2017_Tab.tsv",
#     "2018": "data/NSDUH_2018_Tab.tsv",
#     "2019": "data/NSDUH_2019_Tab.tsv",
#     "2020": "data/NSDUH_2020_Tab.tsv"
}

# Columns to extract
columns = ['QUESTID2', 'SPD_USCR']

# Output folder
output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Process each file
for year, file_path in file_paths.items():
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        
        # Extract specified columns
        filtered_df = df[columns]
        
        # Rename the column to 'SPDYR'
        filtered_df.rename(columns={'SPD_USCR': 'K6SCMON'}, inplace=True)
        
        # Save to new CSV file
        output_file = os.path.join(output_folder, f"spd_{year}.csv")
        filtered_df.to_csv(output_file)
        
        print(f"Processed {year}: Saved to {output_file}")
    except Exception as e:
        print(f"Error processing {year}: {e}")


# In[67]:


import pandas as pd
import os

# File paths for each year
file_paths = {
#     "2002": "data/NSDUH_2002_Tab.tsv",  # Replace with actual file paths
#     "2003": "data/NSDUH_2003_Tab.tsv",
#     "2004": "data/NSDUH_2004_Tab.tsv",
#     "2005": "data/NSDUH_2005_Tab.tsv",
#     "2006": "data/NSDUH_2006_Tab.tsv",
#     "2007": "data/NSDUH_2007_Tab.tsv",
    "2008": "data/NSDUH_2008_Tab.tsv",
    "2009": "data/NSDUH_2009_Tab.tsv",
    "2010": "data/NSDUH_2010_Tab.tsv",
    "2011": "data/NSDUH_2011_Tab.tsv",
    "2012": "data/NSDUH_2012_Tab.tsv",
    "2013": "data/NSDUH_2013_Tab.tsv",
    "2014": "data/NSDUH_2014_Tab.tsv",
    "2015": "data/NSDUH_2015_Tab.tsv",
    "2016": "data/NSDUH_2016_Tab.tsv",
    "2017": "data/NSDUH_2017_Tab.tsv",
    "2018": "data/NSDUH_2018_Tab.tsv",
    "2019": "data/NSDUH_2019_Tab.tsv",
    "2020": "data/NSDUH_2020_Tab.tsv"
}

# Columns to extract
columns = ['QUESTID2', 'K6SCMON']

# Output folder
output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Process each file
for year, file_path in file_paths.items():
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        
        # Extract specified columns
        filtered_df = df[columns]
        
        # Save to new CSV file
        output_file = os.path.join(output_folder, f"spd_{year}.csv")
        filtered_df.to_csv(output_file)
        
        print(f"Processed {year}: Saved to {output_file}")
    except Exception as e:
        print(f"Error processing {year}: {e}")


# In[68]:


import os
import pandas as pd

# Path to the folder containing the files
folder_path = "preprocessed_data"

# List to store individual DataFrames
dataframes = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv") and file_name.startswith("spd_"):
        # Extract the year from the file name
        year = int(file_name.split("_")[1].split(".")[0])
        
        # Load the file into a DataFrame
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, index_col=0)
        
        # Add the 'year' column
        df['year'] = year
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df


# In[ ]:


combined_df['year'].nunique()


# In[2]:


import pandas as pd
import os

# File paths for each year
file_paths = {
    "2005": "data/NSDUH_2005_Tab.tsv",
    "2006": "data/NSDUH_2006_Tab.tsv",
    "2007": "data/NSDUH_2007_Tab.tsv",
    "2008": "data/NSDUH_2008_Tab.tsv",
    "2009": "data/NSDUH_2009_Tab.tsv",
    "2010": "data/NSDUH_2010_Tab.tsv",
    "2011": "data/NSDUH_2011_Tab.tsv",
    "2012": "data/NSDUH_2012_Tab.tsv",
    "2013": "data/NSDUH_2013_Tab.tsv",
    "2014": "data/NSDUH_2014_Tab.tsv",
    "2015": "data/NSDUH_2015_Tab.tsv",
    "2016": "data/NSDUH_2016_Tab.tsv",
    "2017": "data/NSDUH_2017_Tab.tsv",
    "2018": "data/NSDUH_2018_Tab.tsv",
    "2019": "data/NSDUH_2019_Tab.tsv",
    "2020": "data/NSDUH_2020_Tab.tsv"
}

# Columns to extract
columns = ['PRVHLTIN', 'GRPHLTIN', 'QUESTID2']

# Output folder
output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Process each file
for year, file_path in file_paths.items():
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        
        # Extract specified columns
        filtered_df = df[columns].copy()
        
        # Add 'YEAR' column
        filtered_df['YEAR'] = int(year)
        
        # Modify 'QUESTID2' to prepend year
        filtered_df['QUESTID2'] = filtered_df['QUESTID2'].astype(str).apply(lambda x: f"{year}_{x}")
        
        # Save to new CSV file
        output_file = os.path.join(output_folder, f"treat_{year}.csv")
        filtered_df.to_csv(output_file, index=False)
        
        print(f"Processed {year}: Saved to {output_file}")
    except Exception as e:
        print(f"Error processing {year}: {e}")


# In[3]:


pd.read_csv('preprocessed_data/treat_2005.csv')


# In[5]:


import os
import pandas as pd

# Path to the folder containing the files
folder_path = "preprocessed_data"

# List to store individual DataFrames
dataframes = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv") and file_name.startswith("treat_"):
        # Extract the year from the file name
        year = int(file_name.split("_")[1].split(".")[0])
        
        # Load the file into a DataFrame
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df


# In[10]:


import pandas as pd
import numpy as np


# 'PRVHLTIN'
combined_df['PRVHLTIN'] = combined_df['PRVHLTIN'].replace({94: np.nan, 97: np.nan, 98: np.nan})
combined_df['PRVHLTIN'] = combined_df['PRVHLTIN'].astype('category')

# 'GRPHLTIN'
combined_df['GRPHLTIN'] = combined_df['GRPHLTIN'].replace({94: np.nan, 97: np.nan, 98: np.nan, 99:2})
combined_df['GRPHLTIN'] = combined_df['GRPHLTIN'].astype('category')

combined_df


# In[15]:


df = pd.read_csv('data/final_processed.csv', index_col=0)
df = df.drop(['year_x', 'year_y'], axis=1, errors='ignore')
df


# In[16]:


# Merge on 'QUESTID2'
merged_df = pd.merge(df, combined_df, on='QUESTID2', how='inner')  # Use 'left', 'right', or 'outer' if needed

merged_df


# In[17]:


merged_df.to_csv('data/final_processed.csv', index=False)


# In[ ]:




