import os
from argparse import ArgumentParser
from pathlib import Path

from evaluate_program import WebNLG
from main import get_logger

from typing import List, Set, Any
from tqdm import tqdm
import pandas as pd


class IntersctionClustering:
    def __init__(self, min_intersecction=1, max_unique_relations_in_cluster=15, cluster_size_threshold=1) -> None:
        self.min_intersecction = min_intersecction
        self.max_unique_relations_in_cluster = max_unique_relations_in_cluster
        self.cluster_size_threshold = cluster_size_threshold

        self.clusters = set()
        self.cluster_unique_relations = []

    def fit(self, data: List[set]) -> set[tuple[frozenset[str]]]:
        self.clusters = set()
        clusters = self.__cluster(data, set())
        self.__merge_clusters(clusters)
        return self.clusters

    def unique_relations_per_cluster(self, data: List[set]) -> List[set[str]]:
        self.cluster_unique_relations = []
        if self.clusters == set():
            self.fit(data)

        for cluster in self.clusters:
            unique_relations = set()
            for element in cluster:
                unique_relations.update(element)
            self.cluster_unique_relations.append(unique_relations)

        return self.cluster_unique_relations

    def __cluster(self, data: list[set], excluded_relations: set[str]) -> dict[str, list[set]]:
        clusters = dict()

        unique_relations = self.__find_unique_relations(data)
        unique_relations = unique_relations.difference(excluded_relations)
        for relation in unique_relations:
            relations_sets = self.__find_sets_with_relation(data, relation)
            clusters[relation] = relations_sets

        return clusters

    def __find_unique_relations(self, data: list[set]) -> set[str]:
        unique_relations = set()
        for relations in data:
            unique_relations.update(relations)
        return unique_relations

    def __find_sets_with_relation(self, data: list[set], relation: str) -> list[set]:
        sets = []
        for relations in data:
            if relation in relations:
                sets.append(relations)
        return sets

    def __merge_clusters(self, clusters: dict[str, list[set]]) -> None:
        for cluster_partition_relation, cluster in clusters.items():
            unique_relations = self.__find_unique_relations(cluster)
            cluster_size = len(cluster)

            if self.__stop_clustering(unique_relations, cluster_size):
                cluster = tuple(frozenset(x) for x in cluster)
                # if cluster not in self.clusters:
                self.clusters.add(cluster)
            else:
                common_relations = self.__find_common_relations(unique_relations, cluster)
                clusters = self.__cluster(cluster, common_relations)
                self.__merge_clusters(clusters)

    def __stop_clustering(self, unique_relations: set[str], cluster_size: int) -> bool:
        return (len(unique_relations) <= self.max_unique_relations_in_cluster or
                cluster_size <= self.cluster_size_threshold)

    def __find_common_relations(self, unique_relations: set[str], cluster: list[set]) -> set[str]:
        common_relations = set()
        for relation in unique_relations:
            if self.__relation_in_all_sets(relation, cluster):
                common_relations.add(relation)
        return common_relations

    def __relation_in_all_sets(self, relation: str, cluster: list[set]) -> bool:
        for relations in cluster:
            if relation not in relations:
                return False
        return True


def generate_subsets_of_sizes(ground_set, sunsets_sizes: List[int]) -> List[set[Any]]:
    all_subsets = []
    for size in sunsets_sizes:
        all_subsets.extend(generate_subsets_of_size(ground_set, size))
    return all_subsets


def generate_subsets_of_size(ground_set, size: int) -> List[set[Any]]:
    if type(ground_set) is not list:
        ground_set = list(ground_set)
    if size == 0:
        return [set()]
    if size == 1:
        return [{x} for x in ground_set]

    all_subsets = []
    for i in range(len(ground_set)):
        element = ground_set[i]
        rest = ground_set[i + 1:]
        for subset in generate_subsets_of_size(rest, size - 1):
            all_subsets.append({element}.union(subset))
    return all_subsets


def generate_unique_subsets_of_sizes(clusters: List[set[str]], sizes: List[int]) -> Set[frozenset[str]]:
    unique_subsets = set()
    for cluster in tqdm(clusters):
        subsets = generate_subsets_of_sizes(cluster, sizes)
        subsets = [frozenset(subset) for subset in subsets]
        unique_subsets.update(subsets)
    return unique_subsets


def parse_args():
    script_dir = os.path.dirname(__file__)
    default_log_dir = Path(script_dir).parent / 'logs'
    default_output_file = Path(script_dir).parent / 'res' / 'augmentation_relations.csv'

    parser = ArgumentParser()
    parser.add_argument("--log-dir", "-ld", default=default_log_dir, type=str, help="log directory")
    parser.add_argument("--output-file", "-of", default=default_output_file, type=str, help="output file")

    parser.add_argument("--max-unique-relations-in-cluster", "-murc", default=40, type=int, help="max unique relations in cluster")
    parser.add_argument("--cluster-size-threshold", "-cst", default=3, type=int, help="cluster size threshold")
    parser.add_argument("--split", "-s", default='train', type=str, help="dataset split")

    parser.add_argument("--test-set", "-t", default=False, action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger = get_logger(args.log_dir, 'clustering', stream=True, time_in_filename=True)

    webnlg = WebNLG()
    webnlg.load([args.split])
    logger.info(f'Loaded {len(webnlg.data)} samples from {args.split} split of WebNLG dataset')

    df = pd.DataFrame([vars(entry) for entry in webnlg.data])

    # Extract the set of pred elements
    df['pred_elements'] = df['data'].apply(lambda x: tuple(set(sorted([triple.pred for triple in x]))))

    # Flatten the pred_elements column to get all predicates
    all_predicates = [pred for preds_tuple in df['pred_elements'] for pred in preds_tuple]
    unique_predicates = set(all_predicates)
    logger.info(f"Number of unique predicates in {args.split} split: {len(unique_predicates)}")

    grouped_df = df.groupby(['pred_elements'])
    # Group by number of elements and pred elements and select the first row from each group
    one_sample_per_relation_set_df = grouped_df.head(1)
    logger.info(f'unique sets of relations: {one_sample_per_relation_set_df.shape[0]}')

    predicate_sets = one_sample_per_relation_set_df['pred_elements']
    predicate_sets = [set(pred_set) for pred_set in predicate_sets]

    logger.info(f'Clustering ...')
    intersection_clustering = IntersctionClustering(
        max_unique_relations_in_cluster=args.max_unique_relations_in_cluster,
        cluster_size_threshold=args.cluster_size_threshold
    )
    clusters = intersection_clustering.fit(predicate_sets)
    unique_relations_per_cluster = intersection_clustering.unique_relations_per_cluster(predicate_sets)
    logger.info(f'Finished clustering. Number of clusters: {len(clusters)}')

    logger.info(f'Augmenting training set ...')
    augmented_trainig_set = unique_relations_per_cluster
    augmented_training_relations = generate_unique_subsets_of_sizes(augmented_trainig_set, [2, 3, 4])
    augmented_training_relations = [set(x) for x in list(augmented_training_relations)]
    logger.info(f'Number of training samples obtained by augmentation: {len(augmented_training_relations):,}')

    dataset_samples = one_sample_per_relation_set_df['pred_elements']
    dataset_samples = [set(pred_set) for pred_set in dataset_samples]
    unique_dataset_relation_sets = set()
    for relation_set in dataset_samples:
        unique_dataset_relation_sets.add(frozenset(relation_set))

    logger.info(f'Filtering out already existing samples from the augmented training set ...')
    samples_to_generate = []
    for candidate in augmented_training_relations:
        if candidate not in unique_dataset_relation_sets:
            samples_to_generate.append(candidate)

    logger.info(f'Number of samples to generate: {len(samples_to_generate):,}')
    logger.info(
        f'Number of samples not nedded to be generated: {len(augmented_training_relations) - len(samples_to_generate)}')

    if args.test_set:
        webnlg = WebNLG()
        webnlg.load(["test"])
        test_set = [tuple(sorted(set([triple.pred for triple in x.data]))) for x in webnlg.data]
        test_set = [ set (i) for i in set(test_set)]
        logger.info(f"All combinations in test set {len(test_set)}")

        for p in predicate_sets:
            if p in test_set:
                test_set.remove(p)
        logger.info(f"Unknown combinations in test set {len(test_set)}")

        unfiltered_samples_to_generate = samples_to_generate
        samples_to_generate = [p for p in unfiltered_samples_to_generate if p in test_set or
                                any(p.issubset(t) for t in test_set) ]
        logger.info(f"Unknown combinations covered by generated set {len([p for p in unfiltered_samples_to_generate if p in test_set] )}")
        logger.info(f"Number of samples to generate for test_set {len(samples_to_generate)}")

    logger.info(f'Saving samples to generate to {args.output_file}')
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples_to_generate):
            line = ';'.join(list(sample))
            f.write(f'{line}\n')

    logger.info(f'Augmentation relations saved to {args.output_file}')
    logger.info(f'Finished')
