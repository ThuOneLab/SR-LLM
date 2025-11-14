import csv
import re
import ast
from sympy import lambdify
import numpy as np
from typing import List

def process_benchmark(expression):
    pow_regexp = r"pow\((.*?),(.*?)\)"
    pow_replace = r"((\1) ^ (\2))"
    processed = re.sub(pow_regexp, pow_replace, expression)

    div_regexp = r"div\((.*?),(.*?)\)"
    div_replace = r"((\1) / (\2))"
    processed = re.sub(div_regexp, div_replace, processed)
    for i in range(5+1):
        processed = processed.replace(f"x{i}", f"x_{i}")
    return processed


def open_csv(file_name: str):
    equations = []
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            equations.append(
                (
                    row["name"],
                    int(row["var_num"]),
                    process_benchmark(row["expression"]),
                    "U",
                    float(row["lb"]),
                    float(row["ub"]),
                    int(row["sample_num"]),
                )
            )
    return equations


def open_csv_operators_easy(file_name: str):
    equations = []
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader: # csv读取默认是字符串形式读取，不会自动转换为列表变量
            equations.append(
                (
                    row["name"],
                    process_benchmark(row["expression"]),
                    ast.literal_eval(row["prefix_expression"]), #将"['div', 'sub', 'exp', 'mul']"转为['div', 'sub', 'exp', 'mul']
                    int(row["number_of_expression"]),
                    ast.literal_eval(row["operator_list"]),
                    int(row["number_of_operators"]),
                    ast.literal_eval(row["const_list"]),
                    int(row["number_of_const"]),
                )
            )
    # 获取所有元组中第8个元素的最大值
    max_n_const = max([eq[7] for eq in equations])
    print("equations[7] number_of_const 的最大值是:", max_n_const) # 8
    
    return equations


def expr_to_func(sympy_expr, variables: List[str]):
    def cot(x):
        return 1 / np.tan(x)

    def acot(x):
        return 1 / np.arctan(x)

    def coth(x):
        return 1 / np.tanh(x)
    
    def sqrt(x):
        return np.sqrt(np.abs(x))

    return lambdify(
        variables,
        sympy_expr,
        modules=["numpy", {"cot": cot, "acot": acot, "coth": coth}],
    )


def create_dataset(f, 
                   n_var=2, 
                   f_mode = 'col',
                   ranges = [-1,1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0,
                   func_type: str = 'np',
                   noise_level: float = 0.0,
                   distribution: str = 'uniform'):
    '''
    cate dataset
    
    Args:
    -----
        f : function
            the symbolic formula used to cate the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    turns:
    --------
        dataset : dic
            Train/test inputs/labels a dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
         
    Example
    -------
    >>> f = lambda x: np.exp(np.sin(np.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = cate_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    np.Size([100, 2])
    '''

    np.random.seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    if func_type == 'np':
        train_input = np.zeros((train_num, n_var))
        test_input = np.zeros((test_num, n_var))
        for i in range(n_var):
            train_input[:,i] = np.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
            test_input[:,i] = np.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
    elif func_type == 'numpy':
        train_input = np.zeros((train_num, n_var))
        test_input = np.zeros((test_num, n_var))
        for i in range(n_var):
            train_input[:,i] = np.random.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
            test_input[:,i] = np.random.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
                
    train_label = np.zeros((train_num, 1))
    test_label = np.zeros((test_num, 1))
    
    train_label = np.zeros((train_num, 1))
    test_label = np.zeros((test_num, 1))
    for i in range(train_num):
        # 我希望f能够根据n_var的数量来决定输入的数量,而不是直接输入一个向量test_input[i]
        train_label[i] = f(*list(train_input[i]))
    for i in range(test_num):
        test_label[i] = f(*list(test_input[i]))

    y_rms = ((train_label ** 2).mean()) ** 0.5
    epsilon = noise_level * np.random.normal(0, y_rms, len(train_label)).reshape(-1, 1)
    train_label = train_label + epsilon
    
    # if has only 1 dimension
    if len(train_label.shape) == 1:
        train_label = train_label.unsqueeze(dim=1)
        test_label = test_label.unsqueeze(dim=1)
        
    def normalize(data, mean, std):
            return (data-mean)/std
            
    if normalize_input == True:
        mean_input = np.mean(train_input, dim=0, keepdim=True)
        std_input = np.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = np.mean(train_label, dim=0, keepdim=True)
        std_label = np.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset['train_input'] = train_input
    dataset['test_input'] = test_input

    dataset['train_label'] = train_label
    dataset['test_label'] = test_label

    return dataset


def extract_python_code_blocks(markdown_file):
    '''
    extract python code blocks from a markdown file
    '''
    code_blocks = []
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
        code_blocks = []
        code_blocks = re.findall(r'```python(.*?)```', markdown_content, re.DOTALL)
        code_blocks += re.findall(r'```Python(.*?)```', markdown_content, re.DOTALL)
        code_blocks += re.findall(r'``` python(.*?)```', markdown_content, re.DOTALL)
        code_blocks += re.findall(r'``` Python(.*?)```', markdown_content, re.DOTALL)
    return code_blocks


def open_csv_operators_feyn(file_name: str):
    equations = []
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader: # csv读取默认是字符串形式读取，不会自动转换为列表变量
            print(row["name"])
            equations.append(
                (
                    row["name"],
                    process_benchmark(row["expression"]),
                    ast.literal_eval(str(row["prefix_expression"])), #将"['div', 'sub', 'exp', 'mul']"转为['div', 'sub', 'exp', 'mul']
                    int(row["number_of_expression"]),
                    ast.literal_eval(str(row["operator_list"])),
                    int(row["number_of_operators"]),
                    ast.literal_eval(str(row["const_list"])),
                    int(row["number_of_const"]),
                    int(row["number_of_fixed_const"])
                )
            )
    # 获取所有元组中第8个元素的最大值
    max_n_const = max([eq[7] for eq in equations])
    print("equations[7] number_of_const 的最大值是:", max_n_const) # 8
    
    return equations