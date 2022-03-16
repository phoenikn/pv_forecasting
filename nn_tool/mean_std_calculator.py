import torch
import torch.utils.data


def calculate(dataset: torch.utils.data.Dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    data = next(iter(loader))
    mean = torch.mean(data[0], dim=[0, 2, 3, 4])
    std = torch.std(data[0], dim=[0, 2, 3, 4])

    print("Mean of the dataset is:", mean.tolist())
    print("Std of the dataset is:", std.tolist())

    return mean.tolist(), std.tolist()

