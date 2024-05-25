from typing import Dict

from pandas import DataFrame

from GeneticModel.POP import POPULATION

from Models import FileManagement
from Models import DataManagement
import pandas as pd

FILE_PATH: str = "data/iphi2802.csv"
DO_LOGGING = True
DO_PRINT = True

test_values: list = [
    {"pop_size": 20, "cross_rate": 0.6, "mut_rate": 0.0},
    {"pop_size": 20, "cross_rate": 0.6, "mut_rate": 0.01},
    {"pop_size": 20, "cross_rate": 0.6, "mut_rate": 0.1},
    {"pop_size": 20, "cross_rate": 0.9, "mut_rate": 0.01},
    {"pop_size": 20, "cross_rate": 0.1, "mut_rate": 0.01},
    {"pop_size": 200, "cross_rate": 0.6, "mut_rate": 0.0},
    {"pop_size": 200, "cross_rate": 0.6, "mut_rate": 0.01},
    {"pop_size": 200, "cross_rate": 0.6, "mut_rate": 0.1},
    {"pop_size": 200, "cross_rate": 0.9, "mut_rate": 0.01},
    {"pop_size": 200, "cross_rate": 0.1, "mut_rate": 0.01}
]


def import_data() -> tuple[dict[str, int], DataFrame]:
    file_path: str = "data/iphi2802.csv"

    data: pd.DataFrame = FileManagement.Import(file_path).from_csv()

    # keep only data where region_main = 1683
    data = data[data['region_main_id'] == 1683]

    # remove unnecessary columns
    DataManagement.ClearData().from_columns(data, [
        'metadata',
        'id',
        'region_main',
        'region_sub',
        'date_str',
        'date_circa',
        'date_min',
        'date_max',
        'region_main_id',
        'region_sub_id'
    ])

    # clear unnecessary characters
    DataManagement.ClearData().from_characters(
        data,
        r_characters=['\[', '\]', '-', '\.'],
        character='',
        column_names=["text"]
    )

    current_dict = {}
    for idx, word in enumerate(DataManagement.Visualization().get_dictionary(data['text'])):
        current_dict[word[0]] = idx
    return current_dict, data


def main(vocabulary: Dict[str, int], data: pd.DataFrame, execution_num: int,
         population_size: int, crossover_rate: float, mutation_rate: float) -> None:

    inscription = "[---] αλεξανδρος ουδις [---]"
    POPULATION_SIZE = population_size
    CROSSOVER_RATE = crossover_rate
    MUTATION_RATE = mutation_rate

    OUTPUT_FILE = f'./exports/PS_{POPULATION_SIZE}_CR_{CROSSOVER_RATE}_MR_{MUTATION_RATE}.log'
    export = FileManagement.Export(OUTPUT_FILE)
    if DO_PRINT:
        print(f"{'-' * 100}\nrun: {execution_num}\n")
    if DO_LOGGING:
        export.append_to_txt(f"{'-' * 100}\n")
        export.append_to_txt(f"Execution: {execution_num}\n")

    pop = POPULATION(
        population_size=POPULATION_SIZE,
        dictionary=vocabulary,
        data=data['text'],
        inscription=inscription,
        mutation_rate=MUTATION_RATE,
        crossover_probability=CROSSOVER_RATE,
        output=export
    )

    while True:
        # Generate Mating Pool
        pop.rank_selection()
        # Create next Generation and Calculate its Fitness
        pop.evolve()
        # Evaluate Population
        pop.evaluate()

        # Write to log file
        pop.__write_debug_info__(DO_LOGGING)
        # Print Information
        pop.__print_debug_info__(DO_PRINT)

        if pop.__terminate__():
            break

    if DO_LOGGING:
        debug_str = "------------------------------------------------"
        debug_str += f"\nTotal Generations: {pop.generations}\nTotal Population: {POPULATION_SIZE}   ---   Mutation Rate: {MUTATION_RATE}  ---  Crossover Probability: {CROSSOVER_RATE}\nAverage Fitness: {pop.__getAverageFitness__()}  ---  Best Overall Fitness: {pop.best_overall_fitness}\n"
        debug_str += "------------------------------------------------"

        export.append_to_txt(debug_str)

    if DO_PRINT:
        temp_str = "------------------------------------------------"
        temp_str += f"\nTotal Generations: {pop.generations}\nTotal Population: {POPULATION_SIZE}   ---   Mutation Rate: {MUTATION_RATE}  ---  Crossover Probability: {CROSSOVER_RATE}\nAverage Fitness: {pop.__getAverageFitness__()}  ---  Best Overall Fitness: {pop.best_overall_fitness}\n"
        temp_str += "------------------------------------------------"

        print(temp_str)

    print(f"Execution { execution_num + 1 } ENDED")


if __name__ == "__main__":
    vocab, text = import_data()
    num_of_runs = 10

    val = test_values[7]
    for val in test_values:
        for i in range(num_of_runs):
            main(vocab, text, i, val['pop_size'], val['cross_rate'], val['mut_rate'])
