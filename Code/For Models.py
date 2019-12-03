import pandas as pd

# reading csv file data
movie_data_orig = pd.read_csv(r"C:\Users\Madhuri Yadav\Downloads\Final-Project-Group8-master\Final-Project-Group8-master\Code\movies_metadata.csv")
# print(movie_data_orig)     # [45466 rows x 24 columns]

# removing 12 irrelevant columns
df_cleaned = movie_data_orig.drop(["adult", "belongs_to_collection", "homepage", "original_language",
                                   "original_title", "overview", "poster_path", "production_countries",
                                   "spoken_languages", "status", "tagline", "video" ], axis=1)

# print(df_cleaned.columns)   #  'budget', 'genres', 'id', 'imdb_id' 'popularity', 'production_companies', 'release_date', 'revenue', 'runtime', 'title', 'vote_average', 'vote_count']
#df_cleaned.dtypes       # release_date is of object (i.e. string data type) instead of datetime

#Extracting Month from release date
df_cleaned['release_date_temp'] = pd.to_datetime(df_cleaned['release_date'],format='%Y-%m-%d', errors='coerce')  #Converting string to datetime
df_cleaned['release_month'] = pd.to_datetime(df_cleaned['release_date_temp']).dt.month #extracting month from datetime(Releasedate) column
df_cleaned['release_month'] = pd.to_numeric(df_cleaned['release_month'],errors='coerce') #converting float to int
df_cleaned = df_cleaned.drop(['release_date_temp'], axis=1)