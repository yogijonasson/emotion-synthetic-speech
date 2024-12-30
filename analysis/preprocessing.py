import pandas as pd

# Load the files
map_df = pd.read_csv("data/raw/map.csv")
results_df = pd.read_csv("data/raw/results.csv")

# Extract necessary columns from map.csv
map_columns = ['New Name', 'Form question no']
recording_mapping = map_df[map_columns]

# Preprocess the data to long format
long_format_data = []

for index, row in recording_mapping.iterrows():
    recording_name = row['New Name']
    form_question_no = row['Form question no']
    rating_col = 8 + (form_question_no - 1) * 2  # Adjust for zero-indexing
    additional_emotions_col = rating_col + 1
    
    # Extract relevant columns
    temp_df = results_df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, rating_col, additional_emotions_col]]
    temp_df.columns = ['Timestamp', 'Username', 'Consent', 'Name', 'Age', 'Gender', 
                       'Current_Location', 'Country_of_Origin', 
                       f'{recording_name}_rating', f'{recording_name}_additional_emotions']
    
    # Add metadata for long format
    temp_df['Recording'] = recording_name
    temp_df['Rating'] = temp_df[f'{recording_name}_rating']
    temp_df['Additional_Emotions'] = temp_df[f'{recording_name}_additional_emotions']
    
    # Drop the now redundant wide-format columns
    temp_df = temp_df.drop(columns=[f'{recording_name}_rating', f'{recording_name}_additional_emotions'])
    
    # Append to the list for combining later
    long_format_data.append(temp_df)

# Combine all recordings into a single DataFrame
long_format_df = pd.concat(long_format_data, axis=0)

# Split the "Recording" column into Emotion, Pitch, and Naturalness
recording_split = long_format_df['Recording'].str.split('_', expand=True)
long_format_df['Sentiment'] = recording_split[0].str[:-1]
long_format_df['Pitch'] = recording_split[1]
long_format_df['Naturalness'] = recording_split[2]

# Save the reshaped data to a new CSV
output_path = "data/processed/long_format_results_with_split.csv"
long_format_df.to_csv(output_path, index=False)

# Print a preview of the long-format data with the split columns
print("Long-format data with split columns saved to:", output_path)
print(long_format_df.head())
