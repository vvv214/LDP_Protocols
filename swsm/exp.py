import argparse

from data import Data
from svsm import SVSM


def main():
    args.top_k = 32

    data = Data(args)
    finder = SVSM(args, data)
    cand_dict = finder.find()
    print(cand_dict)


parser = argparse.ArgumentParser(description='Comparisor of different schemes.')

parser.add_argument('--top_k', type=int, default=32,
                    help='specify how many to pick up a.k.a., the top k')
parser.add_argument('--epsilon', type=float, default=4,
                    help='specify the differential privacy parameter, epsilon')
args = parser.parse_args()
main()
