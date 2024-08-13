from text_preprocessing import normalize
from multiset import Multiset
from func_timeout import func_timeout, FunctionTimedOut
NO_RULE_ERROR_CODE = -404


class NLGRule:
    def __init__(self, relation_list, rule_code):
        self.relation_list = relation_list
        self.rule_code = rule_code

    def prepare_exec_code(self, triplets):
        pass

    def _order_triples(self, triples):
        # Create a dictionary to map predicates to their order index
        pred_order_dict = {pred: index for index, pred in enumerate(self.relation_list)}
        
        # Sort the triples based on the predicate order
        sorted_triples = sorted(triples, key=lambda triple: pred_order_dict.get(triple.pred, float('inf')))
        
        return sorted_triples
    
    def exec_rule(self, triplets):
        try:
            return func_timeout(5, NLGRule.exec_rule2, args=(self, triplets), kwargs=None)
        except FunctionTimedOut:
            # Handle exceptions
            return '', "Error: Program did not terminate within 5 seconds."     

    def exec_rule2(self, triplets):
        triplets = self._order_triples(triplets)
        result_dict = {}
        combined_script = self.prepare_exec_code(triplets)
        try:
            # Execute the combined script with a custom local namespace
            exec(combined_script, globals(), locals())
            # Get the updated output from the result_dict
            output = result_dict.get('output', '')
            return output, None
        except Exception as e:
            # Handle exceptions
            output = result_dict.get('output', '')
            return output, str(e)


class TemplateRule(NLGRule):
    def __init__(self, relation_list, rule_code):
        self.relation_list = relation_list
        self.template = rule_code

    def prepare_exec_code(self, triplets):
        pass

    def fill_template(self,triple):
        """
        Fills a template with the data from the triple
        """
        output = self.template
        for item, placeholder in [
            (triple.subj, "<subject>"), 
            (triple.pred, "<predicate>"), 
            (triple.obj, "<object>")
        ]:
            output = output.replace(placeholder, normalize(item, 
                    remove_quotes=True, 
                    remove_parentheses=True))
        return output
    
    def exec_rule(self, triplets):
        if len(triplets)> 1 :
            print(f"ERROR: Template rule used for more triples than 1 ({self.relation_list}), {triplets}")
        out = [self.fill_template(triplet) for triplet in triplets ]
        return " ".join(out), None


class Program:
    def __init__(self):
        self.rules = {}

    def add_rule(self, rule):
        relationset_str = self.list_of_rel2dictkey(rule.relation_list) 
        if relationset_str in self.rules:
            print(f"WARN: Replacing an existing rule for {rule.relation_list}")
        self.rules[relationset_str] = rule

    def exec(self, relations, triplets):
        relationset_str = self.list_of_rel2dictkey(relations)
        if relationset_str in self.rules:
            return self.rules[relationset_str].exec_rule(triplets)
        return NO_RULE_ERROR_CODE, f"No rule for a given combination {relations}"

    def process_input(self, relations, triplets):
        out, err = self.exec(relations, triplets)
        if out == NO_RULE_ERROR_CODE:
            known_relations = self.get_known_relations()
            is_in_domain = all(rel in known_relations for rel in relations)
            if not is_in_domain:
                return "OUT OF DOMAIN"
            else:
                result = []
                # print("No rule, split")
                for rel, trip in self._make_split(relations, triplets):
                    out = self.process_input(rel,trip)
                    if not isinstance(out, str):
                        print(rel)
                        print(trip)
                        print(out)
                    result.append(out)
                return " ".join(result)
        return out

    def _make_split(self, relations, triplets):
        available_rules = [Multiset(r) for r in self.rules]
        relations_set = Multiset(relations)
        result = [] 
        while len(relations_set) != 0:
            available_rules = [r for r in available_rules if r.issubset(relations_set)]
            if len(available_rules) == 0:
                print(f"{relations_set}: {available_rules}")
                break
            best_rule = max(available_rules, key=len)
            pred_to_use = best_rule.copy()
            best_triplets = []
            for t in triplets:
                if t.pred in pred_to_use:
                    best_triplets.append(t)
                    pred_to_use = pred_to_use - Multiset([t.pred])
            result.append((best_rule,best_triplets))
            relations_set = relations_set - best_rule
            triplets = [t for t in triplets if t not in best_triplets]
        return result
        
    def has_rule(self, relations):
        relationset_str = self.list_of_rel2dictkey(relations)
        return relationset_str in self.rules

    def get_known_relations(self):
        relations = []
        for rule in self.rules.values():
            relations.extend(rule.relation_list)
        return set(relations)

    def write_program(self, output_dir, name):
        writer = ProgramWriter(output_dir, name)
        for rule in self.rules.values():
            writer.add_rule(rule.relation_list, rule.rule_code)
        writer.add_print_stmt()
        writer.write_program()

    def list_of_rel2dictkey(self, relations):
        return tuple(sorted(relations))

    def add_json_templates(self, templates_filename):
        import json
        with open(templates_filename) as f:
            templates = json.load(f)
        for relation in templates:
            relationName = normalize(relation)
            relationset_str = self.list_of_rel2dictkey([relationName]) 
            if relationset_str not in self.rules:
                rule = TemplateRule([relationName], templates[relation][0])
                self.add_rule(rule)
    def add_fake_json_templates(self, templates_filename):
        import json
        with open(templates_filename) as f:
            templates = json.load(f)
        for relation in templates:
            relationName = normalize(relation)
            relationset_str = self.list_of_rel2dictkey([relationName]) 
            if relationset_str not in self.rules:
                rule = TemplateRule([relationName], "<subject> <predicate> <object>")
                self.add_rule(rule)
class ProgramWriter:
    def __init__(self, output_dir, name):
        self.name = name
        self.output_dir = output_dir
        self.state = "initial"

        self.program = ""
        self.__add_header()

    def __add_header(self):
        # read header from file and add to program
        header_file = self.output_dir / "header.py"
        with open(header_file, "r", encoding="utf-8") as f:
            header = f.read()
            self.program += header

    def add_rule(self, relation_set, rule):
        if_statement = f'''
    if relations == {relation_set}:'''
        self.program += if_statement
        rule = self.adjust_indentation(rule, 8)
        self.program += rule
        self.state = "pending if stmts"

    def adjust_indentation(self, rule, indent_level):
        # Split the rule string into lines
        rule_lines = rule.split("\n")

        existing_indentation = 0
        # Find the first non-empty line and determine its indentation
        for line in rule_lines:
            if line.strip():  # Check if the line is not empty
                existing_indentation = len(line) - len(line.lstrip())
                break

        # Add additional spaces to each line based on the existing indentation
        indented_rule = "\n".join(" " * indent_level + line[existing_indentation:] for line in rule_lines)

        # Add the indented rule to the program
        return indented_rule

    def add_print_stmt(self):
        print_stmt = '''
    print(output)'''
        self.program += print_stmt
        self.state = "initial"

    def write_program(self):
        output_file = self.output_dir / f"{self.name}.py"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(self.program)
