from argparse import ArgumentParser

import dataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument( "--data-dir",default="./data", help="Path to the image")

    args = parser.parse_args()
    dataset.HelmetDataset(args.data_dir)