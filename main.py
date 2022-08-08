"""Main entry to the code"""
from src import UI

def preprocessing():
    """
    Preprocessing step of the pipeline
    """
    ui = UI.UserInterface()
    ui.run_preprocessing()


def train_test():
    """
    Train test step of the pipeline
    """
    ui = UI.UserInterface()
    ui.train_main_model(
        "test",
        "test",
        "sample_data/one_hot_seqs.npy",
        "sample_data/cell_type_array.npy",
        "sample_data/peak_names.npy")


if __name__ == "__main__":
    train_test()
