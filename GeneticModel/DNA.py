import random
from typing import List, Dict

from numpy import mean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .GENE import GENE


class DNA:
    def __init__(self, genes: List[GENE] = None, inscription: List[str] = None, dictionary: Dict[str, int] = None, **kwargs) -> None:
        """A class that represents a DNA of my generic algorithm
        genes: list of genes | None if we want the to be created randomly
        inscription: a list of strings that represent the inscription (we assume that there are 2 missing words at the start and end of the inscription)
        dictionary: a dictionary that contains all words and their corresponding index ex: {"alexander": 0}
        in **kwargs we can pass 2 parameters 'vectorizer' and 'vectorized_data'
        If they are passed then cosine similarity will be calculated in constructor"""
        if inscription is None:
            raise ValueError("Inscription cannot be None")
        self.inscription = inscription

        if dictionary is None:
            ValueError("dictionary cannot be None")
        self.dictionary = dictionary
        self.similarity = None

        if genes is None:
            # randomly create 2 genes
            self.genes = [GENE(word=random.choice(list(dictionary.keys()))) for _ in range(2)]
        else:
            # genes list is given
            if len(genes) != 2:
                raise ValueError("The number of genes must be 2")
            if type(genes[0]) is not GENE:
                raise ValueError("The first given gene is not a type of GENE")
            if type(genes[1]) is not GENE:
                raise ValueError("The second given gene is not a type of GENE")
            self.genes = genes

        # calculate similarity if vectorizer and vectorized data is given
        if 'vectorizer' and 'vectorized_data' in kwargs:
            self.calculate_similarity(kwargs['vectorizer'], kwargs['vectorized_data'])

    def __repr__(self) -> str:
        return f"DNA[{', '.join(str(gene) for gene in self.genes)}]"

    def __get_inscription(self):
        return f"{self.genes[0].word} {' '.join(word for word in self.inscription)} {self.genes[1].word}"

    def get_genes(self) -> List[GENE]:
        return self.genes

    def get_DNA_vector(self, vectorizer: TfidfVectorizer = None) -> List[int]:
        """Convert inscription to a vector using the provided vectorizer,
        vectorizer: the vectorizer used to vectorize data, tested with TfidfVectorizer of sklearn package"""
        if vectorizer is None:
            raise ValueError("You have to specify the vectorizer to create the DNA vector")

        return vectorizer.transform([self.__get_inscription()])

    def calculate_similarity(self, vectorizer: TfidfVectorizer = None, vectorized_data: List[float] = None) -> float:
        """Calculating the similarity between vectorized data and inscription
        vectorizer: the vectorizer used to vectorize data
        vectorized_data: the vectorized data, used to calculate the similarity, tested with TfidfVectorizer from sklearn"""
        if vectorizer is None or vectorized_data is None:
            raise ValueError("You have to specify vectorizer and vectorized data to calculate similarity")

        similarity = mean(cosine_similarity(vectorized_data, self.get_DNA_vector(vectorizer)))
        self.similarity = similarity

    def get_similarity(self) -> float:
        if self.similarity is None:
            raise ValueError("Similarity is not yet calculated. Please run the calculation first.")
        return self.similarity

    def crossover(self, partner):
        if random.random() < 0.5:
            return DNA([self.genes[random.randint(0, 1)], partner.genes[random.randint(0, 1)]], self.inscription, self.dictionary)
        else:
            return DNA([partner.genes[random.randint(0, 1)], self.genes[random.randint(0, 1)]], self.inscription, self.dictionary)

    def mutate(self, mutate_prob) -> None:
        if random.random() < mutate_prob:
            self.genes = [GENE(random.choice(list(self.dictionary.keys()))), GENE(random.choice(list(self.dictionary.keys())))]


if __name__ == '__main__':
    import pandas as pd
    from Models import FileManagement, DataManagement

    file_path: str = "../data/iphi2802.csv"

    data: pd.DataFrame = FileManagement.Import(file_path).from_csv()

    # keep only data where region_main = 1683
    data = data[data['region_main_id'] == 1683]

    # remove unnecessary columns
    columns_to_remove = [
        'metadata', 'id', 'region_main', 'region_sub', 'date_str', 'date_circa',
        'date_min', 'date_max', 'region_main_id', 'region_sub_id']
    DataManagement.ClearData().from_columns(data, columns_to_remove)

    # clear unnecessary characters
    DataManagement.ClearData().from_characters(
        data,
        r_characters=['\[', '\]', '-', '\.'],
        character='',
        column_names=["text"]
    )

    test_dictionary = {}
    for idx, test_word in enumerate(DataManagement.Visualization().get_dictionary(data['text'])):
        test_dictionary[test_word[0]] = idx

    test_vectorizer = TfidfVectorizer(vocabulary=list(test_dictionary.keys()))
    test_vectorized_data = test_vectorizer.fit_transform(data['text'])

    dna = DNA(dictionary=test_dictionary, inscription=['αλεξανδρος', 'ουδις'], vectorizer=test_vectorizer, vectorized_data=test_vectorized_data)

    print(f"DNA before mutation{dna}")
    print(f"DNA's similarity before mutation: {dna.get_similarity()}")
    print()

    dna.mutate(1)
    print(f"DNA after mutation{dna}")
    dna.calculate_similarity(vectorizer=test_vectorizer, vectorized_data=test_vectorized_data)
    print(f"DNA's similarity after mutation: {dna.get_similarity()}")
