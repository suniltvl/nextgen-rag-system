from datasets import load_dataset

def ragbenchloader(dataset_name: str, split: str = "test") -> None:
    datasets = [
        'covidqa', 'cuad', 'delucionqa',
        'emanual', 'expertqa', 'finqa',
        'hagrid', 'hotpotqa', 'msmarco',
        'pubmedqa', 'tatqa', 'techqa'
    ]

    if dataset_name in datasets:
        ragbench_dataset = load_dataset("rungalileo/ragbench", dataset_name, split=split)
        print(type(ragbench_dataset))
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not in the supported list.")


def main():
    print("Hello from nextgen-rag-system!")
    ragbenchloader("finqa")


if __name__ == "__main__":
    main()
