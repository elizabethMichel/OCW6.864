import numpy
import numpy as np


# From Michael Collins' article Probabilistic Context-Free Grammars (PCFGs) of Columbia University.
# http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf
# Implementation of the CKY algorithm for PCFGs on page 14 (July 12th 2024)
def main():
    # Sentence for which we want to find the parse tree.
    sentence = ["the", "man", "saw", "the", "dog", "with", "the", "telescope"]

    # Non-terminal rules and their probability to occur (key is right-side of the rule for easier retrieval).
    non_terminal_rules = {"NPVP": ("SS", 1),
                          "Vi": ("VP", 0.3),
                          "VtNP": ("VP", 0.5),
                          "VPPP": ("VP", 0.2),
                          "DTNN": ("NP", 0.8),
                          "NPPP": ("NP", 0.2),
                          "INNP": ("PP", 1)}

    # Terminal symbols (words) categories and probabilities.
    terminal_rules = {"Vi": [("sleeps", 1)],
                      "Vt": [("saw", 1)],
                      "NN": [("man", 0.1), ("woman", 0.1), ("telescope", 0.3), ("dog", 0.5)],
                      "DT": [("the", 1)],
                      "IN": [("with", 0.6), ("in", 0.4)]}

    # Compute the parse tree.
    cky_algorithm(sentence, terminal_rules, non_terminal_rules)
    inside_algorithm(sentence, terminal_rules, non_terminal_rules)


# Tag words in sentence with grammatical categories from terminal rules dictionary.
def word_categories(s: list, t_rules: dict):
    cat = {}
    for x in t_rules.items():
        for value in x[1]:
            cat[value[0]] = x[0]

    cat_s = []
    for w in s:
        category = cat.get(w, "undefined")
        cat_s.append((w, category))
    return cat_s


# Determine the parse tree of a sentence.
def cky_algorithm(sentence: list, t_rules: dict, nt_rules: dict):
    s_length = len(sentence)
    table, cat_table = init_table(sentence, t_rules)
    subtree = []
    # Length of the span.
    for length in range(1, s_length):
        # Start of the span.
        for i in range(0, s_length - length):
            j = i + length
            max_prob = 0
            max_cat = ""
            candidate = ""
            # Partition of the span : words of the sub-sentence are divided into two groups (y and z).
            for s in range(i, j):
                # Left subtree
                y = cat_table[s, i]
                # Right subtree
                z = cat_table[j, s + 1]

                x, rule_prob = nt_rules.get(y + z, ("", 0))
                current_prob = rule_prob * table[s, i] * table[j, s + 1]

                if current_prob > max_prob:
                    max_prob = current_prob
                    max_cat = x
                    candidate = ((j, i), (s, i), (j, s + 1))

            # Update table on probability and category of the best parse-tree for span.
            table[j, i] = max_prob
            cat_table[j, i] = max_cat
            if candidate != "":
                subtree.append(candidate)
    print_table(sentence, table)
    print_str_table(sentence, cat_table)
    subtree_dict = {x[0]: (x[0], x[1], x[2]) for x in subtree}
    reconstruct_tree(subtree_dict, cat_table, subtree[-1], sentence)


# Determine the probability of a sentence by summing all parse trees.
def inside_algorithm(sentence: list, t_rules: dict, nt_rules: dict):
    s_length = len(sentence)
    table, cat_table = init_table(sentence, t_rules)

    # Length of the span.
    for length in range(1, s_length):
        # Start of the span.
        for i in range(0, s_length - length):
            j = i + length

            span_prob = 0
            max_cat = ""
            max_prod = 0
            # Partition of the span : words of the sub-sentence are divided into two groups (y and z).
            for s in range(i, j):
                x, rule_prob = nt_rules.get(cat_table[s, i] + cat_table[j, s + 1], ("", 0))
                current_prob = rule_prob * table[s, i] * table[j, s + 1]
                # Probabilities are added for every subtree.
                span_prob += current_prob
                if current_prob > max_prod:
                    max_cat = x
                    max_prod = current_prob

            # Update table on probability and category of the best parse-tree for span.
            table[j, i] = span_prob
            cat_table[j, i] = max_cat
    print_table(sentence, table)


# Initialize the square table by sentence's length, and populate its diagonal.
def init_table(s: list, t_rules: dict):
    s_length = len(s)
    table = np.zeros([s_length, s_length])
    cat_table = np.empty([s_length, s_length], dtype='<U5')
    # Categorize words from the sentence.
    cat_sentence = word_categories(s, t_rules)

    # Initialize the table's diagonal (terminal nodes' probabilities).
    for i in range(s_length):
        w, cat = cat_sentence[i]
        # Probability is set to 0 if word cannot be found in the lexicon.
        value = t_rules.get(cat)
        if value is not None:
            table[i, i] = dict(value).get(w, 0)
        else:
            table[i, i] = ""
        cat_table[i, i] = cat

    return table, cat_table


# Pretty-print a float table, associating its columns and rows to the words in the sentence.
def print_table(s: list, table: numpy.array):
    length = len(s)
    if length != len(table):
        return
    max_word = len(max(s, key=len))
    table = np.char.mod('%s', np.round(table, decimals=7)).tolist()

    print('\t ' + ' ' * max_word + '\t '.join(s))
    for i in range(0, length):
        # Print sentence on first row.
        spacing = ' ' * (max_word - len(s[i]))
        print(s[i] + spacing + '\t', '\t\t '.join(table[i]))


# Pretty-print a string table, associating its columns and rows to the words in the sentence.
def print_str_table(s: list, table: np.array):
    length = len(s)
    if length != len(table):
        return
    max_word = len(max(s, key=len))
    table[numpy.where(table == '')] = '--'

    print('\t ' + ' ' * max_word + '\t '.join(s))
    for i in range(0, length):
        # Print sentence on first row.
        spacing = ' ' * (max_word - len(s[i]))
        print(s[i] + spacing + '\t', '\t\t'.join(table[i]))


# Reconstruct tree from pointers saved during parsing, and assign words
# from sentence to terminal nodes.
def reconstruct_tree(t_dict: dict, cat_table: np.array, node: tuple, s: list):
    print(cat_table[node[0]] + "->" + cat_table[node[1]] + " " + cat_table[node[2]])
    left_node = t_dict.get(node[1])
    right_node = t_dict.get(node[2])

    if left_node is not None:
        reconstruct_tree(t_dict, cat_table, left_node, s)
    elif right_node is not None:
        print(s[node[1][0]])
    if right_node is not None:
        reconstruct_tree(t_dict, cat_table, right_node, s)
    elif left_node is not None:
        print(s[node[2][0]])
    if left_node is None and right_node is None:
        print("{0} {1}".format(s[node[1][0]], s[node[2][0]]))


if __name__ == "__main__":
    main()
