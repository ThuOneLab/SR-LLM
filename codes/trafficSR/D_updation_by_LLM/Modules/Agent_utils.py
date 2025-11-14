import os
import numpy as np
import json
import re


delimiter = "####"
interval="            "
def dict2block(dict_obj, block_type='python'):
    dict_str = json.dumps(dict_obj, indent=4)
    markdown_str = f"```{block_type}\n{dict_str}\n```\n"
    return markdown_str


def cosine_distance(vector, matrix):
    """
    :param vector: 1D, (m,1)
    :param matrix: 2D, (m,n)
    :return: numpy array, 1D
    """
    vector, matrix = np.array(vector), np.array(matrix).T
    cosine_similarity = np.zeros(matrix.shape[1])
    dot_product = np.dot(vector, matrix)
    norm_vector = np.linalg.norm(vector)
    norm_matrix = np.linalg.norm(matrix, axis=0)
    cosine_similarity = dot_product / (norm_vector * norm_matrix)

    return cosine_similarity


def extract_python_code_blocks(string):
    '''
    extract python code blocks from a python str
    '''
    code_blocks = []
    code_blocks = re.findall(r'```python(.*?)```', string, re.DOTALL)
    code_blocks += re.findall(r'```Python(.*?)```', string, re.DOTALL)
    code_blocks += re.findall(r'``` python(.*?)```', string, re.DOTALL)
    code_blocks += re.findall(r'``` Python(.*?)```', string, re.DOTALL)
    # assert len(code_blocks) == 1, "There should be only one python code block in this str."
    if len(code_blocks) == 0:
        # raise ValueError("There should be at least one python code block in this str.")
        # raise ValueError("There should be at least one python code block in this str.")
        return ValueError("There should be at least one python code block in this str.")
    code_blocks = [code_blocks[-1]]
    while True:
        if not code_blocks[0].startswith("{"):
            code_blocks[0] = code_blocks[0][1:]
        else:
            break

    return code_blocks[-1]


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def extract_text_between_dollars(pattern, text):
    matches = re.findall(pattern, text)
    # pattern = r'\$\\(.*?)\$'  # for \alpha
    # matches = matches + re.findall(pattern, text)
    return matches


def save_dict_to_text_file(dict_obj, filename):
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(json.dumps(dict_obj, ensure_ascii=False, indent=4) + '\n')
