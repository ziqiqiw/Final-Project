import pandas as pd


def load_remote_huggingface_dataset(dataset_name: str = "David-Egea/Creditcard-fraud-detection") -> tuple:
    from datasets import load_dataset
    ds = load_dataset(dataset_name)
    
    df = pd.DataFrame(ds['train'])  # Convert the dataset to a pandas DataFrame
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def load_local_dataset(file_path: str = "../data/creditcard.csv") -> tuple:
    data = pd.read_csv(file_path)
    X = data.drop("Class", axis=1)
    y = data["Class"]
    return X, y
