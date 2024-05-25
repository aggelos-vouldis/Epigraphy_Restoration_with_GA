import random
from typing import List, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from Models.FileManagement import Export

from numpy import interp, floor

from GeneticModel.DNA import DNA


class POPULATION:
    def __init__(self,
                 population_size: int,
                 dictionary: Dict[str, int],
                 data: pd.Series,
                 mutation_rate: float,
                 crossover_probability: float,
                 output: Export = None,
                 dna_list: List[DNA] = None,
                 inscription: str = None,
                 ) -> None:
        # initialize variables that will be used
        self.best = type(DNA)
        self.terminate_counter = 0
        self.best_overall_fitness = 0
        self.best_fitness = 0
        self.last_best_fitness = 0
        self.finished = False
        self.mating_pool = list()

        # setup other variables
        self.output = output
        self.perfect_score = 1
        self.dictionary = dictionary
        self.vectorizer = TfidfVectorizer(vocabulary=list(dictionary.keys()))
        self.vectorized_data = self.vectorizer.fit_transform(data)

        self.not_changed_inscription = inscription

        self.mutation_rate = mutation_rate
        self.crossover_probability = crossover_probability

        inscription = inscription.replace('[---] ', '')
        inscription = inscription.replace(' [---]', '')
        self.inscription = inscription.split(" ")
        if dna_list is not None:
            self.dna_list = dna_list
            self.population_size = len(dna_list)
        else:
            self.dna_list = self.create_random_population(population_size)
            self.population_size = population_size

        self.generations = 0

    def __repr__(self) -> str:
        return f"POPULATION({', '.join(str(dna) for dna in self.dna_list)}, pop inscription= '{self.not_changed_inscription}')"

    def __len__(self) -> int:
        return self.population_size

    def create_random_population(self, population_size: int) -> list[DNA]:
        """Function to generate a random population of size population_size"""
        # Create a random population of DNAs
        dna_list = []
        for _ in range(population_size):
            dna = DNA(dictionary=self.dictionary, inscription=self.inscription,
                      vectorizer=self.vectorizer, vectorized_data=self.vectorized_data)
            dna_list.append(dna)
        return dna_list

    def get_fittest(self) -> DNA:
        # Return the fittest DNA in the population based on the similarity metric
        fittest_dna = max(self.dna_list, key=lambda dna: dna.similarity)
        return fittest_dna

    # Generate a mating pool
    def rank_selection(self) -> None:
        """Implementation for Fitness Proportionate Selection"""

        # Clear the mating_pool
        self.mating_pool = list()

        # sort the population based on fitness
        pop_sorted = sorted(self.dna_list, key=lambda x: x.similarity)

        # rank the population of the sorted array
        ranked_dna_dict = list()
        for i, item in enumerate(pop_sorted):
            ranked_dna_dict.append({'rank': len(pop_sorted)-i, 'DNA': item})

        for dna_dict in ranked_dna_dict:
            for i in range(dna_dict['rank'] * 10):
                self.mating_pool.append(dna_dict['DNA'])

    def evolve(self) -> None:
        """Refill the generation with children from the mating pool"""
        # clear the past population
        last_population = self.dna_list
        self.dna_list = list()

        # Refill the population with children from the mating pool
        for dna in last_population:
            while True:
                # check if crossover is done
                if self.crossover_probability > random.random():
                    continue
                a = int(floor(random.randint(0, len(self.mating_pool) - 1)))
                b = int(floor(random.randint(0, len(self.mating_pool) - 1)))

                parentA = self.mating_pool[a]
                parentB = self.mating_pool[b]

                child = parentA.crossover(parentB)
                child.mutate(self.mutation_rate)

                self.dna_list.append(child)
                break

        # Calculate the fitness of all members of the new population
        for dna in self.dna_list:
            dna.calculate_similarity(vectorizer=self.vectorizer, vectorized_data=self.vectorized_data)
        self.generations += 1

    def evaluate(self) -> None:
        self.best = self.get_fittest()
        world_record = self.best.similarity

        # change the best fitness and the last best fitness
        self.last_best_fitness = self.best_fitness
        self.best_fitness = world_record

        if self.best_fitness > self.best_overall_fitness:
            self.best_overall_fitness = self.best_fitness

        if self.best_fitness == self.last_best_fitness or (self.best_fitness - self.last_best_fitness) <= 0.01:
            self.terminate_counter += 1
        else:
            self.terminate_counter = 0

        if world_record == self.perfect_score:  # perfect score has reached
            self.finished = True

    def __terminate__(self) -> bool:
        if self.finished:
            return True

        if self.terminate_counter >= 50:
            return True

        if self.generations >= 1000:
            return True
        return False

    def __getAverageFitness__(self) -> float:
        total = 0
        for dna in self.dna_list:
            total += dna.similarity

        return total / len(self.dna_list)

    def __print_debug_info__(self, show: bool) -> None:
        if not show:
            return

        fittest_dna = self.get_fittest()
        best_inscription = f"{fittest_dna.get_genes()[0].get_word()} {self.inscription[0]} {self.inscription[1]} {fittest_dna.get_genes()[0].get_word()}"

        temp_str = ''
        temp_str += (f"Generation {self.generations} " +
                     f"| Average Generation Fitness: {self.__getAverageFitness__()} " +
                     f"| Best Fitness: {self.best_fitness} " +
                     f"| Population Length: {len(self.dna_list)}" +
                     f"| Best DNA: '{best_inscription}'\n")

        print(temp_str)

    def __write_debug_info__(self, write: bool):
        if not write:
            return
        if self.output is None:
            return

        fittest_dna = self.get_fittest()
        best_inscription = f"{fittest_dna.get_genes()[0].get_word()} {self.inscription[0]} {self.inscription[1]} {fittest_dna.get_genes()[0].get_word()}"

        message = (f"Generation {self.generations} " +
                   f"| Average Generation Fitness: {self.__getAverageFitness__()} " +
                   f"| Best Fitness: {self.best_fitness} " +
                   f"| Population Length: {len(self.dna_list)}" +
                   f"| Best DNA: '{best_inscription}'\n")
        self.output.append_to_txt(message)


if __name__ == '__main__':
    # test POP class

    from Models import FileManagement, DataManagement

    file_path: str = "../data/iphi2802.csv"

    d: pd.DataFrame = FileManagement.Import(file_path).from_csv()

    # keep only data where region_main = 1683
    d = d[d['region_main_id'] == 1683]

    # remove unnecessary columns
    columns_to_remove = [
        'metadata', 'id', 'region_main', 'region_sub', 'date_str', 'date_circa',
        'date_min', 'date_max', 'region_main_id', 'region_sub_id']
    DataManagement.ClearData().from_columns(d, columns_to_remove)

    # clear unnecessary characters
    DataManagement.ClearData().from_characters(
        d,
        r_characters=['\[', '\]', '-', '\.'],
        character='',
        column_names=["text"]
    )

    test_dictionary = {}
    for idx, test_word in enumerate(DataManagement.Visualization().get_dictionary(d['text'])):
        test_dictionary[test_word[0]] = idx

    test_vectorizer = TfidfVectorizer(vocabulary=list(test_dictionary.keys()))
    test_vectorized_data = test_vectorizer.fit_transform(d['text'])

    inscr = "[---] αλεξανδρος ουδις [---]"

    MUTATION_RATE = 0.01
    CROSSOVER_RATE = 0.6
    pop = POPULATION(
        population_size=5,
        dictionary=test_dictionary,
        data=d['text'],
        inscription=inscr,
        mutation_rate=MUTATION_RATE,
        crossover_probability=CROSSOVER_RATE
    )

    print(pop)
    # Generate Mating Pool
    pop.rank_selection()
    # Create next Generation and Calculate its Fitness
    pop.evolve()
    # Evaluate Population
    pop.evaluate()
    print(pop)
