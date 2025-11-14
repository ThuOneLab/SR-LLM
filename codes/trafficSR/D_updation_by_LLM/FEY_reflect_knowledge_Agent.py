import copy
import json
import os
import re
import textwrap
import time
import csv

# from langchain.callbacks import OpenAICallbackHandler
from langchain.callbacks import OpenAICallbackHandler
# from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from rich import print
from codes.trafficSR.D_updation_by_LLM.Agent import Agent
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_utils import *
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_default_setting import *
from codes.trafficSR.D_updation_by_LLM.Modules.Expressions_Reflector import Expressions_Reflector_Feynman

class Equation_Info:
    def __init__(self, equation_filename=None,superparent_name=None, symbol_names=None, symbol_descriptions=None, equation_description=None, equation_str=None,prefix_expression=None,constants=None):
        self.equation_filename=equation_filename
        self.superparent_name = superparent_name
        self.symbol_names = symbol_names
        self.symbol_descriptions = symbol_descriptions
        self.equation_description = equation_description
        self.equation_str = equation_str
        self.prefix_expression = prefix_expression
        self.constants = constants


class FEY_reflect_knowledge_AGENT(Agent):
    def __init__(self, trainResultFolder=None, agent_config=DEFAULT_AGENT_CONFIG, port="15732"):
        super(FEY_reflect_knowledge_AGENT, self).__init__(trainResultFolder, agent_config, port)
        self.init_memory()
        self.expression_reflector = Expressions_Reflector_Feynman()

    def init_memory(self):
        system_message = 'You are a physicist with an extensive accumulation of knowledge in physics, capable of reading and interpreting formulas. You will be provided with numerous formulas from different areas of physics, and you need to provide: (1) the meaning and classification of different models in the formula, where you should categorize the symbols in the formula into two categories: "variables" and "constants"; (2) the overall meaning of the symbol combinations in the formula.'
        self.memory = [
            SystemMessage(content=system_message)
        ]

    def process_benchmark(self, expression):
        pow_regexp = r"pow\((.*?),(.*?)\)"
        pow_replace = r"((\1) ^ (\2))"
        processed = re.sub(pow_regexp, pow_replace, expression)

        div_regexp = r"div\((.*?),(.*?)\)"
        div_replace = r"((\1) / (\2))"
        processed = re.sub(div_regexp, div_replace, processed)
        # processed = processed.replace("x1", "x")
        # processed = processed.replace("x2", "y")
        return processed


    def open_csv(self, file_name: str):
        equations = []
        f = open(file_name,'r',encoding='utf8')
        reader = csv.DictReader(f,delimiter=",",)
        for row in reader:
            if len(row["# variables"]) == 0:
                continue
            name = row["\ufeffFilename"]
            formula = row["Formula"].replace("gamma", "Gamma").replace("I", "ii").replace("beta", "Beta")
            num_variables = int(row["# variables"])
            sample_num = 100
            var_info = {}
            for i in range(num_variables):
                var_i_lb = row[f"v{i+1}_low"]
                var_i_ub = row[f"v{i+1}_high"]
                var_i_name = row[f"v{i+1}_name"].replace("gamma", "Gamma").replace("I", "ii").replace("beta", "Beta")
                var_info[var_i_name] = [float(var_i_lb), float(var_i_ub)]
            assert num_variables == len(var_info.keys()), "Number of variables does not match for equation: " + name
            eq = self.process_benchmark(formula)
            var_names = ""
            for i in range(num_variables):
                var_names += list(var_info.keys())[i]
                if i != num_variables - 1:
                    var_names += ","

            equations.append(
                (
                    name,
                    num_variables,
                    eq,
                    var_info,
                    sample_num,
                    var_names
                )
            )
        return equations
    

    def reflect_feynequation_by_llm(self, equation_info: Equation_Info):
        '''
        combine new tokens according to the reward of combinations ouput by SR construction
        '''

        info_dict = {
            "EquationName": equation_info.equation_filename,
            "Formula": equation_info.equation_str,
            "symbols": equation_info.symbol_names,
            "prefix_expression": equation_info.prefix_expression,
        }
        # answer emample
        example_dict = {
            "EquationName": "...(equation name)",
            "Formula": "...(formula expression)",
            "symbols": {
                "symbol_1(symbol name in str format)": {
                    "name": "...(symbol name)",
                    "type": "...(variable or constant or operator)",
                    "description": "...",
                },
                "...": "...",
            },
            "Overall Meaning": "..."
        }
        
        prompt = textwrap.dedent(f"""
            You will be provided with a Python dictionary that stores basic information about a physics formula, including the formula's form and the characters within it. The dictionary is as follows:
            {info_dict}
                                 
            Please first:

            (1) Explain the meaning and classification of different characters in the formula, where you should categorize the characters in the formula into two categories: "variables" and "constants".

            (2) Provide the overall meaning of the character combinations in the formula.

            (3) Organize your output content and present it in the following format:\n""")
        
        prompt += self.dict2block(example_dict)
        prompt += "Please provide the specific Python dictionary containing the physics formula information so I can proceed with the analysis as instructed."

                                
        response = self.talk2llm(prompt, memorize=True, update=False)
        self.add_prompt_old(response, "debug_utils/reflection_feyn.md")

    def reflect_feynequation_prefix_expre(self, equation_info: Equation_Info):
        '''
        combine new tokens according to the reward of combinations ouput by SR construction
        '''

        info_dict = {
            "EquationName": equation_info.equation_filename,
            "Formula": equation_info.equation_str,
            "symbols": equation_info.symbol_names,
            "prefix_expression": equation_info.prefix_expression,
            "constants": equation_info.constants,
        }
        # answer emample
        example_dict = {
            "EquationName": "...(equation name)",
            "Formula": "...(formula expression)",
            "symbols": {
                "symbol_1(symbol name from prefix expression. if it is a constant, then name it as c_1/c_2/c_3...c_n or 1 or pi)": {
                    "type": "...(variable or constant or operator)",
                    "description": "...",
                },
                "...": "...",
            },
            "Overall Meaning": "..."
        }
        
        prompt = textwrap.dedent(f"""
            You will be provided with a Python dictionary that stores basic information about a physics formula, including the formula's form and the characters within it. 
            
            The operator library has a total of ['add', 'mul', 'sub', 'div', 'n4', 'n3', 'n2', 'sqrt', 'sin', 'cos', 'exp', 'log', 'arcsin"], which stand for addition, subtraction, multiplication, division, quadratic, cubic, square, open square, sin, cos, exponential function, logarithmic function, and inverse sine function, respectively.
            
            If there is a corresponding constant term in the prefix expression, it is sorted in the order of the dictionary field 'constants', corresponding to sorting by c_1 c_2 c_3 ... c_n. Unless 1 and pi are present, they are treated as fixed constants and named '1' and 'pi', with the type constant.
            
            The dictionary is as follows. :
            {info_dict}
                                 
            Please first:

            (1) Explain the meaning and classification of different characters in the formula, where you should categorize the characters in the formula into two categories: "variable", "constant" and "operator".

            (2) Provide the overall meaning of the character combinations in the formula.

            (3) Organize your output content and present it in the following format:\n""")
        
        prompt += self.dict2block(example_dict)
        prompt += "Please provide the specific Python dictionary containing the physics formula information so I can proceed with the analysis as instructed."

                                
        response = self.talk2llm(prompt, memorize=True, update=False)
        self.add_prompt_old(response, "debug_utils/reflection_feyn_v2.md")

    def reflect_new_expression(self, expression_dictionary: dict, name: str, save_path="debug_utils/knowledge_feyn_v3.md"):
        # if elite_symbol_dictionary['performance'] != "Good":
        #     raise ValueError("Only good symbol can be reflected.")
        message = self.expression_reflector.reflect_expression2knowledge(expression_dictionary) #原来是reflect_expression
        
        response = ""
        for chunk in self.llm.stream(message):
            response += chunk.content
            print(chunk.content, end="", flush=True)

        self.add_prompt_old(name, save_path)
        self.add_prompt_old('\n', save_path)
        self.add_prompt_old(response, save_path)

        return response



if __name__ == "__main__":
    agent = FEY_reflect_knowledge_AGENT()
    # agent.combine_tokens_by_llm()
    # agent.combine_tokens_by_llm_first_step()
    # agent.talk2llm(prompt="hello",
    #                memorize=False,
    #                update=False)
    # agent.generate_new_symbol("A symbol that represents the acceleration of the following vehicle.")
