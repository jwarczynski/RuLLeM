import re

def extract_triplets(input):
    triplets = input.split('▸')
    # create tuple of (x, relation, y)
    # strip to remove leading and trailing spaces
    triplets = [triplet.split('|') for triplet in triplets]
    return [tuple([x.strip() for x in triplet]) for triplet in triplets]


def extract_relations(input):
    triplets = input.split('▸')
    relations = set()
    for triplet in triplets:
        relation = triplet.split('|')[1].strip()
        relations.add(relation)
    return relations

def normalize(
    s,
    remove_whitespace=True,
    remove_quotes=True,
    remove_underscores=True,
    remove_parentheses=True,
    split_camelcase=True,
):
    if remove_whitespace:
        s = s.strip()

    if remove_underscores:
        s = re.sub(r"_", r" ", s)

    if remove_quotes:
        s = re.sub(r'"', r"", s)
        s = re.sub(r"``", r"", s)
        s = re.sub(r"''", r"", s)

    if remove_parentheses:
        s = re.sub(r"\(", r"", s)
        s = re.sub(r"\)", r"", s)

    if split_camelcase:
        # split basic camel case, lowercase first letters
        s = re.sub(r"([a-z])([A-Z])", lambda m: rf"{m.group(1)} {m.group(2).lower()}", s)

    return s
