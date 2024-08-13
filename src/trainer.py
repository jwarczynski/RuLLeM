import os.path
import os
from argparse import ArgumentParser
from pathlib import Path
import time
import dill
from collections import namedtuple
from pathlib import Path    
from typing import List
from tqdm import tqdm

import ollama

from lm_response_evaluator import SimpleNLGRule, extract_code, get_response_similarity
from program import Program
from logging import getLogger

from text_preprocessing import extract_triplets
from evaluate_program import RDFTriple, WebNLG
from multiset import Multiset

logger = getLogger('trainer')


class ProgramTrainer:

    def __init__(self):
        self.program = Program()

    def train(self, data_dict):
        #interate over data, call construct_rule
        #run enumerate_uknown_combinations and construct_new_combination
        pass

    def construct_rule(self, triplets, reference, sample_id):
        pass

    def enumerate_unknown_combinations(self):
        # perform clustering of relations
        # output what relation combinatinos should be covered by the rules
        pass

    def construct_new_combination(self, relations):
        # generate artificial reference text
        # run construct_rule()
        pass


TemplateTuple = namedtuple("TemplateTuple", ["first_query", "fix_query"])
ChatTuple = namedtuple("ChatTuple", ["role", "content"])


class LMResponseWriter:
    def __init__(self, raw_responses_dir, code_responses_dir, create_dirs=True):
        self.raw_responses_dir = Path(raw_responses_dir)
        self.code_responses_dir = Path(code_responses_dir)

        if create_dirs:
            self._create_dirs_if_not_exist()
        else:
            self.check_dirs_exist()

        self.raw_response_template = "QUERY_NUM: {query_number} - TIME: {time} - DURATION: {duration}s:\n{content}\n\n"
        self.full_response_template = "QUERY_NUM: {query_number} - TIME: {time} - DURATION: {duration}s:\nTRIPLETS: {triplets}\nPROMPT\n{prompt}\nRESP:\n{response}\nCODE{code}\nREFERENCE: {reference}\nRESULT: {rule_result}\nPREPARED:{prepared_code}\n\nMSG:{messages}"
        self.code_responses_template = "QUERY_NUM: {query_number} - TIME: {time} - DURATION: {duration}s: {content}\n\n"

        self.chat_id = None
        self.__query_number = 0

    def new_chat(self, chat_id: str):
        self.chat_id = chat_id
        self.__query_number = 0

    def write_raw_response(self, response: str, duration: int):
        self.__write(
            response,
            duration,
            self.raw_response_template,
            self.raw_responses_dir / f"{self.chat_id}.raw"
        )

    def write_code(self, code: str, duration: int):
        self.__write(
            code,
            duration,
            self.code_responses_template,
            self.code_responses_dir / f"{self.chat_id}.code"
        )

    def write_full_info(self, code: str, duration: int, prompt, triplets, reference, response, messages):
        def execute_rule(triplets, code):
            rule = SimpleNLGRule(triplets, code)
            prepared_code = rule.prepare_exec_code(triplets)
            output, errors = rule.exec_rule(triplets)
            if errors is not None:
                return errors, prepared_code
            return output, prepared_code
        rule_result, prepared_code = execute_rule(triplets, code)
        content = self.full_response_template.format(query_number=self.__query_number,
            time=time.strftime('%Y-%m-%d_%H-%M-%S'),
            duration=duration / 1_000_000_000,
            response=response,
            triplets=triplets,prompt=prompt,code=code, reference=reference, rule_result=rule_result,prepared_code=prepared_code,messages=str(messages))

        with open(self.code_responses_dir / f"{self.chat_id}.full", 'w') as f:
            f.write(content)

    def __write(self, response: str, duration: int, template: str, path: Path):
        response = template.format(
            query_number=self.__query_number,
            time=time.strftime('%Y-%m-%d_%H-%M-%S'),
            duration=duration / 1_000_000_000,
            content=response
        )
        self.__query_number += 1

        with open(path, 'w') as f:
            f.write(response)

    def _create_dirs_if_not_exist(self):
        self.raw_responses_dir.mkdir(parents=True, exist_ok=True)
        self.code_responses_dir.mkdir(parents=True, exist_ok=True)

    def check_dirs_exist(self):
        if not self.raw_responses_dir.exists():
            raise FileNotFoundError(f"Directory {self.raw_responses_dir} does not exist.")
        if not self.code_responses_dir.exists():
            raise FileNotFoundError(f"Directory {self.code_responses_dir} does not exist.")


class EmptyLMResponseWriter(LMResponseWriter):
    def __init__(self, raw_responses_dir, code_responses_dir, create_dirs=True):
        pass

    def new_chat(self, chat_id: str):
        pass

    def write_raw_response(self, response: str, duration: int):
        pass

    def write_code(self, code: str, duration: int):
        pass
    
    def check_dirs_exist(self):
        pass

    def write_full_info(self, code: str, duration: int, prompt, triplets, reference, response, messages):
        pass

class RuleJudge:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def is_ok(self, rule_result, sample_out):
        rule_result = str(rule_result)
        return get_response_similarity(rule_result, sample_out) > self.threshold


class BleuRuleJudge:
    def __init__(self, threshold=0.9):
        import evaluate
        self.threshold = threshold
        self.bleu = evaluate.load("bleu")

    def is_ok(self, rule_result, sample_out):
        value = self.bleu.compute(predictions=[rule_result], references=[sample_out])
        return value["bleu"] > self.threshold


import jellyfish
class JellyRuleJudge:
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def is_ok(self, rule_result, sample_out):
        return jellyfish.jaro_similarity(rule_result,sample_out)> self.threshold


class LevenistainyRuleJudge:
    def __init__(self, threshold=4):
        self.threshold = threshold

    def is_ok(self, rule_result, sample_out):
        rule_result = str(rule_result).lower()
        sample_out = sample_out.lower()
        return jellyfish.damerau_levenshtein_distance(rule_result,sample_out)< self.threshold

class PrecLevenistainyRuleJudge:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def is_ok(self, rule_result, sample_out):
        rule_result = str(rule_result).lower()
        sample_out = sample_out.lower()
        perc = jellyfish.damerau_levenshtein_distance(rule_result,sample_out)/len(sample_out)
        return perc< self.threshold



class LanguageModel:
    def __init__(self, model_name="llama3:70b"):
        self.model_name = model_name

    def query(self, messages, temperature=0.7, seed=None):
        options = {
            "temperature": temperature,
            "seed": seed
        }

        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options=options,
            stream=False
        )

        return response


class EmptyLanguageModel:
    def __init__(self, model_name="gpt-4o"):
        import tiktoken
        self.model_name = model_name
        self.enc = tiktoken.encoding_for_model(model_name)
        self.count = 0

    def query(self, messages, temperature=0.7, seed=None):
        encoded = self.enc.encode(str(messages))
        self.count += len(encoded)
        return "", 0
    

class Prompter:
    def __init__(self, lm: LanguageModel, templates: TemplateTuple, lm_response_writer: LMResponseWriter):
        self._lm = lm
        self.templates = templates
        self.lm_response_writer = lm_response_writer

        self._chat = []

    def new_chat(self, chat_id):
        self._chat = []
        self.lm_response_writer.new_chat(chat_id)

    def ask_for_code(self, triplets, reference) -> str:
        predicates = [triplet.pred for triplet in triplets]
        prompt = self.templates.first_query.format(triplets=triplets, output=reference, relations=predicates)
        return self._query_extract_save(prompt, triplets, reference, temperature=0, seed=None)

    def _query_extract_save(self, prompt, triplets, reference, temperature=0.7, seed=None):
        messages = self.__create_messages(prompt)

        response_json = self._lm.query(messages, temperature=temperature, seed=None)
        response, duration = response_json['message']['content'], response_json['total_duration']
        self._chat.append(ChatTuple(response_json['message']['role'], response_json['message']['content']))
        self.lm_response_writer.write_raw_response(response, duration)

        code = self.__extract_code(response, triplets)
        self.lm_response_writer.write_code(code, duration)
        self.lm_response_writer.write_full_info(code, duration, prompt, triplets, reference, response, messages)

        return code

    def __create_messages(self, prompt):
        self._chat.append(ChatTuple("user", prompt))
        messages = []
        for message in self._chat:
            msg = {
                "role": message.role,
                "content": message.content
            }
            messages.append(msg)

        return messages

    def __extract_code(self, response, relations):
        return extract_code(response, relations)

    def fix_code(self, reference, triplets, rule_result, ):
        prompt = self.templates.fix_query.format(reference, rule_result)
        temp = len(self._chat) / 10 / 2
        return self._query_extract_save(prompt, triplets, reference, temperature=temp, seed=None)


class SimpleProgramTrainer(ProgramTrainer):
    def __init__(self, lm: LanguageModel, templates: TemplateTuple, lm_response_writer: LMResponseWriter,
                 prompter: Prompter, judge: RuleJudge, max_fix_prompts=5, test_set=False):
        super().__init__()
        self.lm = lm
        self.templates = templates
        self.lm_response_writer = lm_response_writer
        self.prompter = prompter
        self._judge = judge
        self.max_fix_prompts = max_fix_prompts
        self.test_set = None
        if test_set:
            webnlg = WebNLG()
            webnlg.load(["test"])
            self.test_set = [Multiset([triple.pred for triple in x.data]) for x in webnlg.data]

    def train(self, dataset, checkpoint_file):
        curr_iteration = -1
        if os.path.isfile(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                self.program = dill.load( f)
                curr_iteration = dill.load(f)
        ile = 0
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
            if i < curr_iteration:
                continue
            predicates = [triplet.pred for triplet in sample.data]
            if self.test_set is not None:
                predicates_set = Multiset(predicates)
                if not any([predicates_set.issubset(t) for t in self.test_set]):
                    continue
            ile += 1
            if not self.program.has_rule(predicates):
                self.construct_rule(sample.data, sample.refs[0], sample.entry_id)
            else:
                print(f"SKIP: Rule already exists {predicates}")
            if i % 100 == 0:
                with open(checkpoint_file, 'wb') as f:
                    dill.dump(self.program, f)
                    dill.dump(i, f)
        #After finishing training
        with open(checkpoint_file, 'wb') as f:
                    dill.dump(self.program, f)
                    dill.dump(i, f)
        print(f"Examples processed {ile}")

    def construct_rule(self, triplets: List[RDFTriple], reference: str, sample_id: str):
        def execute_rule(triplets, code):
            rule = SimpleNLGRule(triplets, code)
            output, errors = rule.exec_rule(triplets)
            if errors is not None:
                return errors, False
            return output, self._judge.is_ok(output, reference)

        self.prompter.new_chat(sample_id)
        fix_query_count = 0

        code = self.prompter.ask_for_code(triplets, reference)
        rule_result, is_rule_ok = execute_rule(triplets, code)
        # print("==== New rule")
        # print(f"{is_rule_ok}\n{triplets}\n{reference}\n{rule_result}\n-----\n{code}\n")
        # print("====")

        while not is_rule_ok and fix_query_count < self.max_fix_prompts:
            self.prompter.lm_response_writer.new_chat(sample_id+f"_fix{fix_query_count}")
            if fix_query_count % 3 == 2:
                self.prompter.new_chat(sample_id+f"_fix{fix_query_count}")
                code = self.prompter.ask_for_code(triplets, reference)
            else:
                code = self.prompter.fix_code(reference, triplets, rule_result)
            rule_result, is_rule_ok = execute_rule(triplets, code)
            # print("==== Error correction")
            # print(f"{is_rule_ok}\n{reference}\n{rule_result}\n-----\n{code}\n")
            # print("====")
            fix_query_count += 1

        if is_rule_ok:
            self.program.add_rule(SimpleNLGRule([triplet.pred for triplet in triplets], code))
        else:
            logger.error(
                f'[ObjectProgramTrainer] '
                f'Failed to generate rule for {triplets} after {fix_query_count} attempts. Skipping...'
            )

def parse_args():
    script_dir = os.path.dirname(__file__)
    parser = ArgumentParser()
    parser.add_argument("--out-file", type=str, help="saved model")
    parser.add_argument("--model", type=str, help="LLM model", default="llama3")
    parser.add_argument("--template", type=str, help="LLM model", default="2")
    parser.add_argument("--levy",  help="LLM model", default=False, action="store_true")
    parser.add_argument("--test-set",  help="LLM model", default=False, action="store_true")
    parser.add_argument("--augment",  help="LLM model", default=False, action="store_true")
    parser.add_argument("--outname",  default="")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    script_path = os.path.dirname(os.path.realpath(__file__))
    lm_raw_responses_dir = Path(script_path) / '..' / 'out2' / f'train{args.outname}' / 'lm_raw_responses'
    lm_code_responses_dir = Path(script_path) / '..' / 'out2' / f'train{args.outname}' / 'lm_code_responses'
    first_query_template_path = Path(script_path) / '..' / 'res' / 'prompt_templates' / f'template{args.template}.txt'
    fix_query_template_path = Path(script_path) / '..' / 'res' / 'prompt_templates' / 'wrong_output_template2.txt'
    output_program_dir = Path(script_path) / '..' / 'out2' / f'train{args.outname}'
    output_program_name = 'rule_program'



    with open(first_query_template_path, 'r') as f:
        first_query_template = f.read()
    with open(fix_query_template_path, 'r') as f:
        fix_query_template = f.read()

    templates = TemplateTuple(first_query=first_query_template, fix_query=fix_query_template)
    lm_response_writer = EmptyLMResponseWriter(lm_raw_responses_dir, lm_code_responses_dir, create_dirs=True)
    lm = LanguageModel(model_name=args.model)
    prompter = Prompter(lm, templates, lm_response_writer)
    if args.levy:
        judge = LevenistainyRuleJudge(6)
    else:
        judge = RuleJudge()

    trainer = SimpleProgramTrainer(lm, templates, lm_response_writer, prompter, judge, 7, args.test_set)
    dataset = WebNLG()
    dataset.load(['train'])
    if args.augment:
        from evaluate_program import AugmentedDataset  
        dataset = AugmentedDataset()
        dataset.load("path.json")
        
    trainer.train(dataset.data, args.out_file)
    print("====")
    print(lm.count)
    trainer.program.write_program(output_program_dir, output_program_name)
    




