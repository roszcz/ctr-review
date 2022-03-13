import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt

import argparse

argument_parser = argparse.ArgumentParser()
data_path_help = """
    data_path: location of the CSV of the clickthrough data
        downloaded from https://www.kaggle.com/c/avazu-ctr-prediction
        has to include columns: `click` and `hour`
"""
argument_parser.add_argument("--data_path", required=True, type=str, help=data_path_help)
sampling_percentage_help = """
    sampling_percentage: percentage value from range (0, 100]. Because the original
        dataset is large you can downsample it to make everything run faster. Default
        value 5 means that only 5%% of the dataset will be used (random sampling every
        1,000,000 rows).
"""
argument_parser.add_argument("--sampling_percentage", type=int, default=5, help=sampling_percentage_help)


# YYMMDDHH
TIME_FORMAT = '%y%m%d%H'


def read_sampled_data(path: str, sampling_percentage: int) -> pd.DataFrame:
    chunksize = 1_000_000
    # 5% should be enough
    samples_per_chunk = chunksize * sampling_percentage // 100

    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        part = chunk.sample(samples_per_chunk)
        parts.append(part)
        if len(part) % 50 == 0:
            print('Got to another one!', dt.datetime.now().isoformat())

    df = pd.concat(parts)

    return df


def load_data(path: str) -> pd.DataFrame:
    max_rows = 1_000_000
    df = pd.read_csv(path, max_rows)
    return df


def inject_features(df: pd.DataFrame) -> pd.DataFrame:
    def read_time(time_token: int) -> dt.datetime:
        time_string = str(time_token)
        return dt.datetime.strptime(time_string, TIME_FORMAT)

    df['timestamp'] = df.hour.apply(read_time)
    return df


def prepare_ctr_frame(path: str, sampling_percentage: int) -> pd.DataFrame:
    df = read_sampled_data(path, sampling_percentage)
    df = inject_features(df)
    ctr = df.resample('H', on='timestamp').click.mean()
    ctr_frame = pd.DataFrame(ctr)
    ctr_frame = ctr_frame.rename(columns={'click': 'ctr'})

    return ctr_frame


def find_outliers(ctr_frame: pd.DataFrame) -> pd.DataFrame:
    ctr_frame['rolling_std'] = ctr_frame.ctr.rolling(window=5, center=True).std()
    ctr_frame['rolling_mean'] = ctr_frame.ctr.rolling(window=5, center=True).mean()
    ids = np.abs(ctr_frame.ctr - ctr_frame.rolling_mean) >= ctr_frame.rolling_std * 1.5
    ctr_frame['is_outlier'] = ids

    return ctr_frame


def outlier_plot(ctr_frame: pd.DataFrame) -> None:
    plt.figure(figsize=[13, 5])
    ids = ctr_frame.is_outlier

    plt.plot(ctr_frame.ctr, '--', color='black', alpha=0.5)
    plt.plot(ctr_frame[ids].ctr, 'o', ms=9, color='tomato', label='CTR Outlier')
    plt.plot(ctr_frame[~ids].ctr, 'o', color='teal', label='CTR Within Norm')

    low_bound = ctr_frame.rolling_mean - 1.5 * ctr_frame.rolling_std
    high_bound = ctr_frame.rolling_mean + 1.5 * ctr_frame.rolling_std
    plt.fill_between(ctr_frame.index, low_bound, high_bound, color='deepskyblue', alpha=0.5, label='Moving Average')

    plt.plot(ctr_frame.rolling_mean, color='deepskyblue')
    plt.ylabel('CTR')
    plt.xlabel('Time')
    plt.title('CTR Outliers Detection')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('plots/outliers.png')


def ctr_over_time_plot(ctr_frame: pd.DataFrame) -> None:
    plt.figure(figsize=[13, 5])
    plt.plot(ctr_frame.ctr, color='black', lw=2, alpha=0.8)
    plt.plot(ctr_frame.ctr, 'o', color='teal', ms=9, alpha=0.5, label='Hour Average')
    plt.ylabel('CTR')
    plt.xlabel('Time')
    plt.legend()
    plt.grid()
    plt.title('CTR Over Time')
    plt.tight_layout()
    plt.savefig('plots/ctr.png')


def main():
    args = argument_parser.parse_args()
    data_path = args.data_path
    sampling_percentage = args.sampling_percentage

    ctr_frame = prepare_ctr_frame(data_path, sampling_percentage)
    ctr_frame = find_outliers(ctr_frame)

    ctr_over_time_plot(ctr_frame)
    outlier_plot(ctr_frame)

    print('done!')


if __name__ == '__main__':
    main()
