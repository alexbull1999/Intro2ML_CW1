import numpy as np


def print_tree(node, depth=0):
    # Indentation based on depth for readability
    indent = "  " * depth

    if node.is_leaf():
        # Print leaf node details
        print(f"{indent}Leaf: Label={node.label}")
    else:
        # Print internal node details
        print(f"{indent}Node: Feature={node.feature}, Threshold={node.threshold}")

        # Recursively print children
        for i, child in enumerate(node.children):
            print(f"{indent}  Child {i}:")
            print_tree(child, depth + 2)


