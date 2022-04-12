import pandas as pd
from matplotlib import pyplot as plt


def plot_loss(csv_dir):
    loss_file = pd.read_csv(csv_dir)
    loss_file = loss_file[2:]
    plt.plot(loss_file["epoch"], loss_file["train_loss"], loss_file["epoch"], loss_file["val_loss"])
    plt.legend(["train_loss", "val_loss"])
    plt.show()


if __name__ == "__main__":
    plot_loss("../convLSTM/training_records_300epochs_8stack_numeric.csv")
