import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize

# reading csv file data
movie_data_orig = pd.read_csv("movies_metadata.csv")

# all column names
print(movie_data_orig.columns)

# removing irrelevant columns
df_cleaned = movie_data_orig.drop(["adult", "belongs_to_collection", "homepage", "id", "imdb_id", "original_language",
                                   "original_title", "overview", "poster_path", "production_countries", "release_date",
                                   "spoken_languages", "status", "tagline", "video" ], axis=1)
len(df_cleaned)       # 45466
# print(df_cleaned.head(5))

# taking into account only those movies where budget and/or revenue is greater than $100,000 (some of values in budget & revenue
# columns are 0, 1, 2, 3 etc which do not make any sense

df_cleaned.dtypes             # datatype of all columns in the dataframe; (budget is of "object" datatype instead of "int")
df_cleaned.describe()           # giving summary statistics of columns with dtype = float64
# budget column contains alpha-numeric characters, so need to fix it
df_cleaned['budget'] = df_cleaned['budget'].str.extract('(\d+)', expand=False)   # removing all non-numeric values from budget column
len(df_cleaned)       # 45466
df_cleaned["budget"] = df_cleaned["budget"].astype(float).fillna(0.0)    # changing budget column from object to float
df_cleaned.dtypes       # confirming that budget's dtype has changed to float64
df_cleaned.describe()
df_cleaned = df_cleaned.loc[df_cleaned['budget'] > 100000]   # subsetting df to only movies with budget greater than $100,000
len(df_cleaned)     # 8298

df_cleaned = df_cleaned.loc[df_cleaned['revenue'] > 1000]      # subsetting df to only movies with revenue greater than $1000
len(df_cleaned)      # 5249
df_cleaned.describe()

# creating our target/label column showing status i.e success/flop movie.
df_cleaned["status"] = df_cleaned["revenue"]/df_cleaned["budget"]
df_cleaned.describe()
df_cleaned.dtypes
# Our criteria for success is any value greater than 1 else flop
df_cleaned["new_status"] = np.nan      # creating a new empty column
df_cleaned["new_status"] = df_cleaned["new_status"].mask( df_cleaned["status"] > 1, 1)
df_cleaned["new_status"] = df_cleaned["new_status"].mask( df_cleaned["status"] <= 1, 0)




# 1. converting (genre) json column to normal string column

# the correct JSON format uses double quotes instead of single quotes
df_cleaned["genres"] = df_cleaned["genres"].str.replace("\'", "\"")
# converting json column to list
k = df_cleaned["genres"].apply(lambda row: pd.DataFrame(json.loads(row))).tolist()
# print(type(k))       <class 'list'>
# print(k[0])     indices + all ids + all genres in first row

# extracting only the major i.e. where "index=0" genre type for each row
df_temp = pd.concat(k)    # creating a df on basis of json list
# df_temp.index        [0, 1, 2, 0, 1, 2, 0, 1, 0, 1, and so on
new_df = df_temp.loc[df_temp.index.isin([0])]

#???????
# len(df_cleaned)     # 5249
# len(new)       # 5238  means there are some missing values in genre column
# df_cleaned.isnull().sum()      # returns the total # of missing/NaN values in each col of df, surprisingly for genres it is 0

# print(new_df["name"])    confirming the major genre values

# fixing "ValueError: cannot reindex from a duplicate axis"
df_cleaned = df_cleaned.reset_index(drop=True)
new_df = new_df.reset_index(drop=True)

# updating genres column with only one major category
df_cleaned["genres"] = new_df["name"]




# 2. converting (production_companies) json column to normal string column






# there are many entries where the number of people who voted for a movie are 1, 2 , 3 etc
df_cleaned = df_cleaned.loc[df_cleaned['vote_count'] > 100]      # subsetting df to only movies where atleast 100 people voted for a movie
len(df_cleaned)      # 3763
