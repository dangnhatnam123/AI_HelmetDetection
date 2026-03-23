from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument( "--image-dir",default="./data", required=True, help="Path to the image")

    args = parser.parse_args()
    print(args.image_dir)