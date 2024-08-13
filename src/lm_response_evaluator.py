import difflib
import re
from program import NLGRule


class SimpleNLGRule(NLGRule):
    def prepare_exec_code(self, triplets):
        code = remove_redundant_code(self.rule_code, self.relation_list)
        code = remove_indents(code)

        combined_script = f"""
# Triplets
from collections import namedtuple
RDFTriple = namedtuple("RDFTriple", ["subj", "pred", "obj"])
relations = [triplet.pred for triplet in triplets]
triplets = {triplets}
# Initialize output variable
output = ""
# Code
{code}
# Return the output
result_dict['output'] = output
"""
        return combined_script


def get_response_similarity(response, reference_text):
    return difflib.SequenceMatcher(None, response, reference_text).ratio()


def extract_code(response, relations):
    # check if <code> tag is present in the response
    if '<code>' not in response:
        #try to extract code from ```
        if '```' in response:
            code = response.split('```')[1]
        else:
            # TODO: handle this case
            return None
    else:
        # Extract the code from the <code> tag
        code = response.split('<code>')[1].split('</code>')[0]

    # remove `print(output) from the code
    code = code.replace("print(output)", "")
    code = remove_redundant_code(code, relations)

    return code


def evaluate_response(triplets, code, reference_text, relations):
    # Combine triplets, code, and reference text into a single Python script
    # remove indents from each line if the code in first line is indented

    code = remove_redundant_code(code, relations)
    code = remove_indents(code)

    combined_script = f"""
# Triplets
triplets = {triplets}
# Initialize output variable
output = ""
# Code
{code}
# Return the output
result_dict['output'] = output
"""
    # with open('../res/combined_scripts/combined_script.py', 'w') as f:
    #     f.write(combined_script)

    result_dict = {}
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


def remove_redundant_code(code, relations):
    def eliminate_re(pattern, code):
        regex = re.compile(pattern, re.MULTILINE)

        # Remove the matching line from the code
        return regex.sub("", code)

    ''' Example usage:
    code = """
    if relations == {'LIVES_IN', 'HAS'}:
        print("This line should be removed")
    print("This line should remain")
    """
    relations = {'LIVES_IN', 'HAS'}
    new_code = remove_redundant_code(code, relations)
    print(f'new_code:\n{new_code}')
    '''

    if code is None:
        return None

    if not isinstance(relations, set):
        relations = set(relations) #set([triple[1] for triple in relations])
    # Define the pattern to match
    pattern = fr"^\s*if\s+\(?relations\s*==\s*.*{re.escape(str(relations))}\)?\s*:.?$"
    # pattern = fr"^\s*if\s+\(?relations\s*==\s*.*:.?$"
    
    # Remove the matching line from the code
    new_code = eliminate_re(fr"^\s*triplets = \[.*$", code)
    new_code = eliminate_re(fr"^\s*python\s*$", new_code)
    new_code = eliminate_re(pattern, new_code)
    

    return new_code


def remove_indents(code):
    if code is None:
        return None
    # Remove leading empty lines
    code_lines = code.split("\n")
    while code_lines and not code_lines[0].strip():
        code_lines.pop(0)

    # Determine the indentation of the first non-empty line
    if code_lines:
        indent = len(code_lines[0]) - len(code_lines[0].lstrip())
    else:
        indent = 0

    # Remove leading indentation from each line
    if indent > 0:
        code_lines = [line[indent:] for line in code_lines]

    return "\n".join(code_lines)
