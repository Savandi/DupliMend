import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess_event_log(file_path, infrequent_threshold=0.01):
    df = pd.read_csv(file_path)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns detected: {categorical_cols}")

    for col in categorical_cols:
        freq = df[col].value_counts(normalize=True)
        frequent_values = freq[freq >= infrequent_threshold].index.tolist()

        df[col] = df[col].apply(lambda x: x if x in frequent_values else 'Other')

        df[col] = pd.Categorical(df[col])

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    numerical_df = df.drop(columns=categorical_cols)
    final_df = pd.concat([numerical_df, encoded_df], axis=1)

    return final_df


if __name__ == "__main__":
    input_file = "../homonym_mend/generated_event_log_homonyms_interleaved_2000.csv"
    output_file = "generated_event_log_homonyms_interleaved_preprocessed_event_log_2000.csv"

    preprocessed_df = preprocess_event_log(input_file, infrequent_threshold=0.01)
    preprocessed_df.to_csv(output_file, index=False)

    print(f"Preprocessed log saved to: {output_file}")
