import random
from typing import List


class GENE:
    def __init__(self, word: str) -> None:
        """A class that represents a gene of my generic algorithm
        word: word of current gene"""

        self.word: str = word

    def __repr__(self) -> str:
        return f"GENE('{self.word}')"

    def get_word(self) -> str:
        return self.word


# testing gene
if __name__ == "__main__":
    test_dictionary = dictionary = {
        'alejandro': 0,
        'udi': 1,
        'syria': 2,
        'greater': 3,
        'east': 4,
        'and': 5,
        'the': 6,
        'west': 7,
        'north': 8,
        'south': 9,
    }
    test_gene = GENE(random.choice(list(test_dictionary.keys())))
    print(test_gene)
