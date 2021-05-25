import os
import pandas as pd
import numpy as np

def make_dummy_data(col_names,
                    col_ranges,
                    save_path,
                    n_data=100):
    '''
    Create dummy data and save to CSV for functional testing purposes

    Arg(s):
        col_names : list[str]
            names of each column
        col_ranges : list[(float, float)]
            list of (min, max) pairs of values for each column
        save_path : str
            path to save csv file to
        n_data : int
            number of rows of data to generate. Default is 100

    Returns: pd.dataframe
        Dataframe representing the dummy data
    '''
    assert len(col_names) == len(col_ranges)

    df = pd.DataFrame()

    # Generate dataframe with random values
    for col_name, (min_val, max_val) in zip(col_names, col_ranges):
        df[col_name] = (np.random.rand(n_data) * (max_val - min_val)) + min_val

    # save dataframe as csv
    df.to_csv(save_path, index=False)

def generate_train_test_data(col_names,
                             col_ranges,
                             save_dir,
                             n_train_data=100,
                             n_test_data=30):
    '''
    Create dummy data and save to CSV for functional testing purposes

    Arg(s):
        col_names : list[str]
            names of each column
        col_ranges : list[(float, float)]
            list of (min, max) pairs of values for each column
        save_dir : str
            directory to save train and test data to
        n_train_data : int
            number of rows of data to generate for training
        n_test_data : int
            number of rows of data to generate for testing

    Returns:
        None
    '''
    if len(col_names) != len(col_ranges):
        raise ValueError('col_names and col_ranges lengths of {} and {} do not match'.format(len(col_names), len(col_ranges)))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    make_dummy_data(
        col_names=col_names,
        col_ranges=col_ranges,
        save_path=os.path.join(save_dir, "train_data.csv"),
        n_data=n_train_data)

    make_dummy_data(
        col_names=col_names,
        col_ranges=col_ranges,
        save_path=os.path.join(save_dir, "test_data.csv"),
        n_data=n_test_data)

if __name__ == "__main__":
    col_names = [
        "bit_rate",
        "prev_bit_rate",
        "buffer_size",
        "bandwidth_throughput_0",
        "bandwidth_throughput_1",
        "bandwidth_throughput_2",
        "bandwidth_throughput_3",
        "bandwidth_throughput_4",
        "bandwidth_throughput_5",
        "bandwidth_throughput_6",
        "bandwidth_throughput_7",
        "bandwidth_time_0",
        "bandwidth_time_1",
        "bandwidth_time_2",
        "bandwidth_time_3",
        "bandwidth_time_4",
        "bandwidth_time_5",
        "bandwidth_time_6",
        "bandwidth_time_7",
        "next_chunk_sizes_0",
        "next_chunk_sizes_1",
        "next_chunk_sizes_2",
        "next_chunk_sizes_3",
        "next_chunk_sizes_4",
        "next_chunk_sizes_5",
        "n_chunks_remaining"
        ]

    col_ranges = [
        # bit rate (cur & previous)
        (0,1),
        (0,1),
        # buffer size
        (0, 10),
        # bandwidth throughput
        (0, 1000),
        (0, 1000),
        (0, 1000),
        (0, 1000),
        (0, 1000),
        (0, 1000),
        (0, 1000),
        (0, 1000),
        # bandwidth time
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        # next chunk sizes
        (0, 1000),
        (0, 1000),
        (0, 1000),
        (0, 1000),
        (0, 1000),
        (0, 1000),
        # n_chunks remaining
        (0,1)]
    save_dir = "dummy_data"
    generate_train_test_data(
        col_names=col_names,
        col_ranges=col_ranges,
        save_dir=save_dir
    )
    # make_dummy_data(
    #     col_names=col_names,
    #     col_ranges=col_ranges,
    #     save_path=save_path)