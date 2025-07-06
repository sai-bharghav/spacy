# #
# # Python script to download datasets from the Hugging Face Hub
# #
# # This script uses the 'datasets' library to easily download and inspect
# # a dataset. It has been updated to download the BBQ Gender Bias dataset.
# #

# # ==============================================================================
# # Step 1: Installation
# # ==============================================================================
# # Before running, you need to install the 'datasets' library, and 'pandas'
# # which is very useful for handling the data.
# #
# # Open your terminal or command prompt (and activate your virtual environment)
# # and run the following command:
# #
# # pip install datasets pandas
# #
# # ==============================================================================

# import pandas as pd
# from datasets import load_dataset
# import os

# def download_and_explore_dataset(dataset_name, split_name, config_name=None):
#     """
#     Downloads a specific configuration and split of a dataset from Hugging Face,
#     prints some examples, and saves it to a CSV file.

#     Args:
#         dataset_name (str): The name of the dataset on Hugging Face.
#         split_name (str): The data split to use (e.g., 'train', 'test', 'validation').
#         config_name (str, optional): The specific configuration or subset to load. Defaults to None.
#     """
#     print(f"Attempting to download dataset: '{dataset_name}'...")
#     if config_name:
#         print(f"Using configuration: '{config_name}'")


#     try:
#         # This is the core command to download the data.
#         # It streams the data and caches it locally for future use.
#         dataset = load_dataset(dataset_name, name=config_name, split=split_name)
#         print("\n✅ Dataset downloaded successfully!")

#     except Exception as e:
#         print(f"\n❌ An error occurred while downloading the dataset.")
#         print(f"Error details: {e}")
#         print("Please check the dataset name and your internet connection.")
#         return

#     # --- Let's explore the data ---
#     print(f"\n--- Exploring the first 3 examples from the '{split_name}' split ---")

#     # The dataset object behaves like a list of dictionaries.
#     for i in range(3):
#         if i < len(dataset):
#             example = dataset[i]
#             print(f"\n--- Example {i+1} ---")
#             print(f"  ID: {example.get('example_id')}")
#             print(f"  Context: {example.get('context')}")
#             print(f"  Question: {example.get('question')}")
#             print(f"  Label (Answer Index): {example.get('label')}")
#             # You can uncomment the line below to see all data for an example
#             # print(f"  Full Data: {example}")

#     # --- Let's save the data to a file ---
#     print("\n--- Converting and saving data to CSV ---")

#     try:
#         # The 'datasets' library integrates well with pandas
#         df = dataset.to_pandas()
        
#         # Create a clean filename from the dataset name
#         clean_dataset_name = dataset_name.replace('/', '_')
#         output_filename = f"{clean_dataset_name}_{split_name}.csv"
        
#         df.to_csv(output_filename, index=False)

#         print(f"\n✅ Success! Data has been saved to '{output_filename}' in your current directory.")

#     except Exception as e:
#         print(f"\n❌ An error occurred while saving the file: {e}")


# if __name__ == "__main__":
#     # --- Configuration ---
#     # Updated to download the BBQ Gender Bias dataset from the provided link.
#     # The dataset name is 'SLLMBias/qa_BBQ_bi_gender'.
    
#     DATASET_TO_DOWNLOAD = 'SLLMBias/cont_stereoset'
#     # CORRECTED: This specific dataset only has a 'validation' split available.
#     SPLIT_TO_DOWNLOAD = 'validation' 

#     download_and_explore_dataset(DATASET_TO_DOWNLOAD, SPLIT_TO_DOWNLOAD)



import os
try:
    from datasets import load_dataset
except ImportError:
    print("The 'datasets' library is not installed.")
    print("Please install it using: pip install datasets")
    exit()

try:
    import pandas as pd
except ImportError:
    print("The 'pandas' library is not installed.")
    print("Please install it using: pip install pandas")
    exit()

try:
    import pyarrow
except ImportError:
    print("The 'pyarrow' library is not installed. It's needed to write Parquet files.")
    print("Please install it using: pip install pyarrow")
    exit()


def download_and_save_bbq_gender_dataset(save_path=".", chunk_size_mb=20):
    """
    Downloads the SLLMBias/qa_BBQ_bi_gender dataset, partitions it into
    chunks of a specified size, and saves each chunk as a Parquet file.

    This function requires the 'datasets', 'pandas', and 'pyarrow' libraries.
    You can install them with: pip install datasets pandas pyarrow

    Args:
        save_path (str): The directory where the files will be saved.
                         Defaults to the current directory.
        chunk_size_mb (int): The approximate target size for each file chunk in MB.
    """
    # The name of the dataset on the Hugging Face Hub
    dataset_name = "SLLMBias/qa_BBQ_bi_gender"
    
    # The base name for the output Parquet files
    base_filename = "qa_BBQ_bi_gender"
    
    print(f"Attempting to download '{dataset_name}' from the Hugging Face Hub...")
    print("This may take a moment...")
    
    try:
        # Load the dataset from the Hugging Face Hub
        dataset = load_dataset(dataset_name, split='validation')
        print("\n✅ Dataset loaded successfully from the Hugging Face Hub.")

        # Convert the dataset to a pandas DataFrame
        print("Converting to DataFrame for processing...")
        df = dataset.to_pandas()
        print(df.shape)
        print(df.head())

        # --- Logic to partition the DataFrame into chunks ---
        
        # Calculate the total memory usage of the DataFrame in bytes
        # total_size_bytes = df.memory_usage(deep=True).sum()
        # target_chunk_size_bytes = chunk_size_mb * 1024 * 1024

        # if total_size_bytes < target_chunk_size_bytes:
        #     # If the total size is less than the chunk size, save as a single file
        #     print(f"Dataset is smaller than {chunk_size_mb}MB. Saving as a single Parquet file.")
        #     full_path = os.path.join(save_path, f"{base_filename}.parquet")
        #     df.to_parquet(full_path, engine='pyarrow', index=False)
        #     print(f"\n✅ Success! Dataset saved to: {os.path.abspath(full_path)}")
        # else:
        #     # Calculate the number of rows that fit into one chunk
        #     bytes_per_row = total_size_bytes / len(df)
        #     rows_per_chunk = int(target_chunk_size_bytes / bytes_per_row)
            
        #     if rows_per_chunk == 0:
        #         rows_per_chunk = 1 # Ensure at least one row per chunk

        #     print(f"Partitioning data into chunks of ~{rows_per_chunk} rows each...")
            
        #     # Loop through the DataFrame and save chunks
        #     num_chunks = 0
        #     for i in range(0, len(df), rows_per_chunk):
        #         chunk_df = df.iloc[i:i + rows_per_chunk]
        #         num_chunks += 1
                
        #         # Define the filename for the current chunk
        #         chunk_filename = f"{base_filename}_part_{num_chunks}.parquet"
        #         full_path = os.path.join(save_path, chunk_filename)
                
        #         # Save the chunk to a Parquet file
        #         chunk_df.to_parquet(full_path, engine='pyarrow', index=False)
        #         print(f"  - Saved chunk {num_chunks} to {chunk_filename}")

        #     print(f"\n✅ Success! Dataset saved in {num_chunks} Parquet files in the directory: {os.path.abspath(save_path)}")

    except Exception as e:
        # Handle potential errors during download or processing
        print(f"\n❌ An unexpected error occurred.")
        print("Please ensure you have 'datasets', 'pandas', and 'pyarrow' installed.")
        print("You can install them using: pip install datasets pandas pyarrow")
        print(f"\nError details: {e}")

if __name__ == "__main__":
    # You can specify a different directory here if you want to save the files elsewhere.
    # For example: 
    # output_directory = "my_datasets"
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    # download_and_save_bbq_gender_dataset(save_path=output_directory, chunk_size_mb=20)
    
    download_and_save_bbq_gender_dataset()