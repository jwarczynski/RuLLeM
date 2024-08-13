import logging
import evaluate
import numpy as np

from collections import defaultdict, namedtuple
from datasets import load_dataset
from text_preprocessing import normalize


logger = logging.getLogger(__name__)

RDFTriple = namedtuple("RDFTriple", ["subj", "pred", "obj"])


class DataEntry:
    """
    An entry in the dataset
    """

    def __init__(self, data, refs, data_type, entry_id, align=None, num_ref_sentences=None, category=None, dialhist=None):
        self.data = data
        self.refs = refs
        self.data_type = data_type
        self.align = align
        self.num_ref_sentences = num_ref_sentences
        self.category = category
        self.dialhist = dialhist
        self.entry_id = entry_id.replace("/", "_")

    def __repr__(self):
        return str(self.__dict__)


class WebNLG:
    """
    The WebNLG dataset: https://gem-benchmark.com/data_cards/web_nlg
    Contains RDF triples from DBPedia and their crowdsourced verbalizations.
    """

    name = "webnlg"

    def __init__(self, *args, **kwargs):
        self.data = []

    def load(self, splits, path=None):
        # load the dataset from HF datasets
        dataset = load_dataset("gem", "web_nlg_en")

        for split in splits:
            data = dataset[split if split != "dev" else "validation"]

            for example in data:
                triples = example["input"]
                triples = [t.split("|") for t in triples]
                triples = [(normalize(x, remove_parentheses=False) for x in t) for t in triples]
                triples = [RDFTriple(*t) for t in triples]

                if split == "test":
                    refs = example["references"]
                else:
                    refs = [example["target"]]

                entry = DataEntry(
                    data=triples, refs=refs, data_type="triples",
                    entry_id=example['webnlg_id'], category=example["category"]
                )
                self.data.append(entry)


class AugmentedDataset:
    def __init__(self, *args, **kwargs):
        self.data = []

    def load(self, path):
        def get_known_relations():
            from evaluate_program import WebNLG
            data = WebNLG()
            data.load(['test', "train"])

            triplets = [dataEntry.data for dataEntry in data.data]
            triplets = [t.pred  for sets in triplets for t in sets]
            known_relations = set(triplets)
            return known_relations
        
        def convert2triple(input, known_relations):
            if len(input) == 1:
                input = input[0].split(",")
            input = [i.strip() for i in input]
            rel = [i for i,j in enumerate(input) if j in known_relations]
            if len(rel) != 1:
                print("ERROR!")
                print(input)
                print(rel)
                return None
            i = rel[0]
            return [", ".join(input[:i]), input[i], ", ".join(input[i+1:])]
            
        import json
        # load the dataset from HF datasets
        with open(path, "r") as f:
            data = json.load(f)

        known_relations = get_known_relations()
        for id, example in  enumerate(data["not_augmented_samples"]):
            triples = example["in"]
            # print(id)
            import re
            triples = re.findall(r'\(.*?\)', triples)
            triples = [i[1:-1] for i in triples]
            # print(triples)
            triples = [convert2triple(t.split("|"), known_relations) for t in triples]
            if any([t is None for t in triples]):
                print(triples)
                print(f"SKIP {id}")
                continue
            # print(triples)
            triples = [(normalize(x, remove_parentheses=False) for x in t) for t in triples]
            
            triples = [RDFTriple(*t) for t in triples]
            if "out" not in example:
                print(f"WARN: Incomplete example {example}")
                continue
            entry = DataEntry(
                data=triples, refs=[example["out"]], data_type="triples",
                entry_id=str(id), category="augmented"
            )
            self.data.append(entry)



# METRICS ==============================

class SingleReferenceMetric:
    def __init__(self) -> None:
        self.name = "SingleReferenceMetric"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        pass

    def compute(self, preds, refs, ref_lens, is_out_domain):
        results = self.eval(preds, refs, ref_lens, is_out_domain)

        i = 0
        merged_results = []
        for len_r in ref_lens:
            merged_results.append(results[i:i + len_r].mean())
            i += len_r

        results = np.array(merged_results)
        print(
            f"{self.name}: {results.mean()} +- {results.std()}; OOD: {results[is_out_domain].mean()}; InD: {results[~is_out_domain].mean()}")


class MultiReferenceMetric:
    def __init__(self) -> None:
        self.name = "MultiReferenceMetric"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        pass

    def compute(self, preds, refs, ref_lens, is_out_domain):
        results = self.eval(preds, refs, ref_lens, is_out_domain)
        print(f"{self.name}: {results} ")


class BLEURT(SingleReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bleurt", module_type="metric")
        self.name = "BLEURT"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs)
        # results_in = self.metric.compute(predictions=[p for i,p in enumerate(preds) if not is_out_domain[i]], references=[r for i,r in enumerate(refs) if not is_out_domain[i]])
        return np.array(results["scores"])


class BERTScore(SingleReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bertscore")
        self.name = "BERTScore"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs, lang="en")
        # results_in = self.metric.compute(predictions=[p for i,p in enumerate(preds) if not is_out_domain[i]], references=[r for i,r in enumerate(refs) if not is_out_domain[i]], lang="en")
        return np.array(results["f1"])


class BLEU(MultiReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bleu")
        self.name = "BLEU"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs)        
        results_in = self.metric.compute(predictions=[p for i,p in enumerate(preds) if not is_out_domain[i]], references=[r for i,r in enumerate(refs) if not is_out_domain[i]])
        good = [(i,j) for i,j in zip(preds,refs) if i not in ("SPLIT NEEDED", "OUT OF DOMAIN")]
        results_good = self.metric.compute(predictions=[i for i,j in good], references=[j for i,j in good])
        return np.array(results["bleu"]), np.array(results_in["bleu"]), np.array(results_good["bleu"]), len(good)/sum(is_out_domain)
    
class METEOR(MultiReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("meteor")
        self.name = "METEOR"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs)
        results_in = self.metric.compute(predictions=[p for i,p in enumerate(preds) if not is_out_domain[i]], references=[r for i,r in enumerate(refs) if not is_out_domain[i]])
        #results_in = self.metric.compute(predictions=preds[~is_out_domain], references=refs[~is_out_domain])
        return np.array(results["meteor"]), np.array(results_in["meteor"])


# EVAL ==============================

def get_basic_metrics():
    return [METEOR(), BLEU()]

from tqdm import tqdm

def evaluate_program(program, metrics):
    data = WebNLG()
    data.load(['test'])

    refs_single = []
    preds_single = []
    refs_multi = []
    preds_multi = []
    ref_lens = []
    is_out_domain = []
    for dataEntry in tqdm(data.data):
        is_out_domain.append(dataEntry.category in ["Film", "MusicalWork", "Scientist"])

        relations = tuple(sorted([i.pred for i in dataEntry.data]))
        # input = [tuple([triplet.subj, triplet.pred, triplet.obj]) for triplet in dataEntry.data]
        output = program.process_input(relations, dataEntry.data)
        if output == "OUT OF DOMAIN":
            is_out_domain[-1] = True

        refs_multi.append(dataEntry.refs)
        preds_multi.append(output)
        for reference_text in dataEntry.refs:
            refs_single.append(reference_text)
            preds_single.append(output)
        ref_lens.append(len(dataEntry.refs))
    is_out_domain = np.array(is_out_domain)

    for metric in metrics:
        if isinstance(metric, MultiReferenceMetric):
            metric.compute(preds_multi, refs_multi, ref_lens, is_out_domain)
        else:
            metric.compute(preds_single, refs_single, ref_lens, is_out_domain)
    return preds_multi, refs_multi


