from torchaudio.datasets import SPEECHCOMMANDS
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required = True, help='path to save files')
    args = parser.parse_args()

    traindataset = SPEECHCOMMANDS(root = args.save_dir, download = True)




