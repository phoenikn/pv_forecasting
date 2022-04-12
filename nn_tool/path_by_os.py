import os

WIN_ROOT = "../"
LINUX_SCRATCH = "/scratch/itee/uqsxu13/nowcasting_feature_data"
INDEX_FOLDER = "index"
DATA_ROOT = "extracted_data"


def get_path(file_dir: str, win_root: str = WIN_ROOT, linux_scratch: str = LINUX_SCRATCH,
             index_folder=INDEX_FOLDER, data_root=DATA_ROOT):
    if ".csv" in file_dir:
        path = get_index(index_folder, file_dir)
    else:
        path = get_data(data_root, file_dir)
    return os.path.join(win_root if os.name == "nt" else linux_scratch, path)


def get_index(index_folder, index_file):
    return os.path.join(index_folder, index_file)


def get_data(data_root, data_folder):
    return os.path.join(data_root, data_folder)
