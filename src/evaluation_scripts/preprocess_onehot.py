import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess_event_log(file_path, infrequent_threshold=0.01):
    # Load CSV
    df = pd.read_csv(file_path)

    # Step 1: Detect categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns detected: {categorical_cols}")

    # Step 2: Compress categorical columns using pandas.Categorical
    for col in categorical_cols:
        # Calculate relative frequencies
        freq = df[col].value_counts(normalize=True)
        frequent_values = freq[freq >= infrequent_threshold].index.tolist()

        # Replace infrequent with 'Other'
        df[col] = df[col].apply(lambda x: x if x in frequent_values else 'Other')

        # Convert to memory-efficient categorical dtype
        df[col] = pd.Categorical(df[col])

    # Step 3: One-hot encode
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    # Step 4: Combine with original numerical data
    numerical_df = df.drop(columns=categorical_cols)
    final_df = pd.concat([numerical_df, encoded_df], axis=1)

    return final_df


# === Example usage ===
if __name__ == "__main__":
    input_file = "../homonym_mend/generated_event_log_homonyms_interleaved_2000.csv"  # Replace with file path
    output_file = "generated_event_log_homonyms_interleaved_preprocessed_event_log_2000.csv"

    preprocessed_df = preprocess_event_log(input_file, infrequent_threshold=0.01)
    preprocessed_df.to_csv(output_file, index=False)

    print(f"✅ Preprocessed log saved to: {output_file}")
