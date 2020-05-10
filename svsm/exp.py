from data import Data
from svsm import SVSM


def main():
    data = Data(dataname='kosarak', limit=20000)
    finder = SVSM(data, top_k=32, epsilon=4)
    cand_dict = finder.find()
    print(cand_dict)


main()
