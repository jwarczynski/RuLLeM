from argparse import ArgumentParser
import os
import random

from pathlib import Path
from typing import List

from lm_response_evaluator import SimpleNLGRule, extract_code
from program import Program
from logging import getLogger

from evaluate_program import RDFTriple, WebNLG
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from trainer import ChatTuple, EmptyLMResponseWriter, LanguageModel, LevenistainyRuleJudge, SimpleProgramTrainer, SimpleNLGRule, TemplateTuple, LMResponseWriter, RuleJudge

logger = getLogger('vllm_trainer')


class VLanguageModel():
    def __init__(self, model_name="neuralmagic/Meta-Llama-3-70B-Instruct-FP8"):
        self.llm = LLM(model=model_name,  max_model_len=3500,
                       gpu_memory_utilization=0.95, max_num_seqs=50, tensor_parallel_size=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_prompts(self, prompters):
        result = [prompt.messages for prompt in prompters]
        text_batch = self.tokenizer.apply_chat_template(
            result, tokenize=False, add_generation_prompt=True)
        return text_batch

    def query(self, pompters, temperature=0.7, seed=None):
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=5000)
        prompts = self.prepare_prompts(pompters)
        outputs = self.llm.generate(prompts, sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs


class DisposablePrompter:
    def __init__(self, templates: TemplateTuple, lm_response_writer: LMResponseWriter):
        self.templates = templates
        self.lm_response_writer = lm_response_writer
        self.triplets = None
        self.reference = None
        self.disabled = True
        self._chat = []
        self.rule_result = None
        self.messages = []

    def create_empty_copy(self):
        return DisposablePrompter(self.templates, self.lm_response_writer)

    def new_chat(self, chat_id):
        self._chat = []
        self.lm_response_writer.new_chat(chat_id)

    def set_example(self, triplets, reference):
        self.triplets = triplets
        self.reference = reference
        self.disabled = False

    def ask_for_code_before(self):
        predicates = [triplet.pred for triplet in self.triplets]
        prompt = self.templates.first_query.format(
            triplets=self.triplets, output=self.reference, relations=predicates)
        self.__create_messages(prompt)

    def ask_for_code_after(self, response):
        self._chat.append(ChatTuple("assistant", response))
        self.lm_response_writer.write_raw_response(response, 0)

        predicates = [triplet.pred for triplet in self.triplets]
        code = self.__extract_code(response, predicates)

        self.lm_response_writer.write_code(code, 0)
        self.lm_response_writer.write_full_info(
            code, 0, "prompt", self.triplets, self.reference, response, self.messages)

        return code

    def __create_messages(self, prompt):
        self._chat.append(ChatTuple("user", prompt))
        self.messages = []
        for message in self._chat:
            msg = {
                "role": message.role,
                "content": message.content
            }
            self.messages.append(msg)

    def __extract_code(self, response, relations):
        return extract_code(response, relations)

    def fix_code_before(self):
        prompt = self.templates.fix_query.format(
            self.reference, self.rule_result)
        self.__create_messages(prompt)


class BatchedProgramTrainer(SimpleProgramTrainer):
    def __init__(self, lm: LanguageModel, templates: TemplateTuple, lm_response_writer: LMResponseWriter,
                 prompter: DisposablePrompter, judge: RuleJudge, max_fix_prompts=5, test_set=False):
        super().__init__(lm, templates, lm_response_writer,
                         prompter, judge, max_fix_prompts, test_set)
        self.prompters = []

    def construct_rule(self, triplets: List[RDFTriple], reference: str, sample_id: str):
        def execute_rule(triplets, code, reference):
            rule = SimpleNLGRule(triplets, code)
            output, errors = rule.exec_rule(triplets)
            if errors is not None:
                return errors, False
            return output, self._judge.is_ok(output, reference)

        new_prompter = self.prompter.create_empty_copy()
        new_prompter.new_chat(sample_id)
        new_prompter.set_example(triplets, reference)
        self.prompters.append(new_prompter)

        if len(self.prompters) > 100:
            print(f"Old size: {len(self.prompters)}")
            for prompter in self.prompters:
                prompter.ask_for_code_before()
            outputs = self.lm.query(self.prompters, temperature=0.0)
            for prompter, output in zip(self.prompters, outputs):
                code = prompter.ask_for_code_after(output)
                rule_result, is_rule_ok = execute_rule(
                    prompter.triplets, code, prompter.reference)
                prompter.rule_result = rule_result
                if is_rule_ok:
                    self.program.add_rule(SimpleNLGRule(
                        [triplet.pred for triplet in prompter.triplets], code))
                    prompter.disabled = True

            fix_query_count = 0
            self.prompters = [
                prompter for prompter in self.prompters if not prompter.disabled]
            print(f"New size: {len(self.prompters)}")
            while len(self.prompters) != 0 and fix_query_count < self.max_fix_prompts:
                for prompter in self.prompters:
                    prompter.lm_response_writer.new_chat(
                        sample_id+f"_fix{fix_query_count}")
                    if fix_query_count % 3 == 2:
                        prompter.new_chat(sample_id+f"_fix{fix_query_count}")
                        code = prompter.ask_for_code_before()
                    else:
                        code = prompter.fix_code_before()
                temp = fix_query_count / 10
                outputs = self.lm.query(self.prompters, temperature=temp)
                for prompter, output in zip(self.prompters, outputs):
                    code = prompter.ask_for_code_after(output)
                    rule_result, is_rule_ok = execute_rule(
                        prompter.triplets, code, prompter.reference)
                    prompter.rule_result = rule_result
                    if is_rule_ok:
                        self.program.add_rule(SimpleNLGRule(
                            [triplet.pred for triplet in prompter.triplets], code))
                        prompter.disabled = True
                self.prompters = [
                    prompter for prompter in self.prompters if not prompter.disabled]
                print(f"New size: {len(self.prompters)}")

                fix_query_count += 1

            for prompter in self.prompters:
                logger.error(
                    f'[ObjectProgramTrainer] '
                    f'Failed to generate rule for {prompter.triplets} after {fix_query_count} attempts. Skipping...'
                )
            self.prompters = []


def parse_args():
    script_dir = os.path.dirname(__file__)
    parser = ArgumentParser()
    parser.add_argument("--out-file", type=str, help="saved model")
    parser.add_argument("--template", type=str, help="LLM model", default="2")
    parser.add_argument("--levy",  help="LLM model",
                        default=False, action="store_true")
    parser.add_argument("--test-set",  help="LLM model",
                        default=False, action="store_true")
    parser.add_argument("--augment",  help="LLM model",
                        default=False, action="store_true")
    parser.add_argument("--outname",  default="")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    script_path = os.path.dirname(os.path.realpath(__file__))
    lm_raw_responses_dir = Path(
        script_path) / '..' / 'out2' / f'train{args.outname}' / 'lm_raw_responses'
    lm_code_responses_dir = Path(
        script_path) / '..' / 'out2' / f'train{args.outname}' / 'lm_code_responses'
    first_query_template_path = Path(
        script_path) / '..' / 'res' / 'prompt_templates' / f'template{args.template}.txt'
    fix_query_template_path = Path(
        script_path) / '..' / 'res' / 'prompt_templates' / 'wrong_output_template2.txt'
    output_program_dir = Path(script_path) / '..' / \
        'out2' / f'train{args.outname}'
    output_program_name = 'rule_program'

    with open(first_query_template_path, 'r') as f:
        first_query_template = f.read()
    with open(fix_query_template_path, 'r') as f:
        fix_query_template = f.read()

    templates = TemplateTuple(
        first_query=first_query_template, fix_query=fix_query_template)
    lm_response_writer = EmptyLMResponseWriter(
        lm_raw_responses_dir, lm_code_responses_dir, create_dirs=True)
    lm = VLanguageModel()
    prompter = DisposablePrompter(templates, lm_response_writer)
    if args.levy:
        judge = LevenistainyRuleJudge(6)
    else:
        judge = RuleJudge()

    trainer = BatchedProgramTrainer(
        lm, templates, lm_response_writer, prompter, judge, 7, args.test_set)
    dataset = WebNLG()
    dataset.load(['train'])
    if args.augment:
        from evaluate_program import AugmentedDataset
        dataset = AugmentedDataset()
        dataset.load(
            "path.json")


    random.shuffle(dataset.data)
    trainer.train(dataset.data, args.out_file)
    print("====")
    print(lm.count)
    trainer.program.write_program(output_program_dir, output_program_name)
