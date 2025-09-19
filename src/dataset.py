from datasets import load_dataset, DatasetDict, load_from_disk
import os

def get_tokenized_dataset(dataset_name: str, dataset_config: str, tokenizer, sequence_length: int, max_samples: int = None):
    """
    Loads, tokenizes, and caches a dataset.
    Cache path includes dataset + tokenizer + vocab size + seq length to avoid stale loads.
    """
    is_main_process = os.environ.get('LOCAL_RANK', '0') == '0'

    tokenizer_name = tokenizer.name_or_path.replace('/', '_')
    vocab_size = len(tokenizer)
    processed_dir_name = f"{dataset_name}_{dataset_config}_{tokenizer_name}_v{vocab_size}_seq{sequence_length}"
    processed_path = os.path.join("processed_datasets", processed_dir_name)

    # Optional: set tokenizer max length to keep warnings quiet
    try:
        tokenizer.model_max_length = sequence_length
    except Exception:
        pass

    if os.path.exists(processed_path):
        if is_main_process:
            print(f"Loading pre-processed dataset from: {processed_path}")
        tokenized_datasets = load_from_disk(processed_path)
    else:
        if is_main_process:
            print(f"No pre-processed dataset found. Processing '{dataset_name}' from scratch...")
        
        raw_datasets = DatasetDict()
        raw_datasets['train'] = load_dataset(dataset_name, dataset_config, split='train')
        raw_datasets['validation'] = load_dataset(dataset_name, dataset_config, split='validation')

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=sequence_length, padding="max_length")

        if is_main_process:
            print("Tokenizing train and validation splits...")
        
        tokenized_datasets = raw_datasets.map(
            tokenize_function, batched=True, remove_columns=["text"],
            num_proc=os.cpu_count()
        )

        if is_main_process:
            print(f"Saving processed dataset to: {processed_path}")
            os.makedirs(processed_path, exist_ok=True)
            tokenized_datasets.save_to_disk(processed_path)

    if max_samples and 'train' in tokenized_datasets:
        if is_main_process:
            print(f"Subsampling training data to {max_samples} examples.")
        tokenized_datasets['train'] = tokenized_datasets['train'].select(range(max_samples))

    return tokenized_datasets