

def generate_reduce_dataset(dataset, digit: int):
    return dataset[dataset.label == digit]
