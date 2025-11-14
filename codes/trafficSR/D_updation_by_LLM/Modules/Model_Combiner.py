import textwrap
from langchain.schema import HumanMessage, SystemMessage

from codes.trafficSR.D_updation_by_LLM.Modules.Agent_utils import *
from codes.trafficSR.D_updation_by_LLM.Modules.defaults import *

from rich import print
import copy

delimiter = "####"


class Model_Combiner():
    def __init__(self):
        pass

    def initialize_system_message_to_combine_model(self):
        self.system_message_to_combine_model = textwrap.dedent(
            f'''
            You are required to act as a helpful assistant, you can provide guidance for constructing a complex car-following model.
            You should not only consider operator rules, but also unit constraints and physical meanings.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat until you decide to output your final answer>
            Response to user:
            {delimiter} <Your final output>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message_to_combine_model = self.system_message_to_combine_model.replace("            ", "")

    def initialize_human_message_to_combine_model(self, library_info: dict = DEFAULT_LIBRARY_INFO):
        # new_library_info = copy.deepcopy(library_info)
        # for name, info in library_info.items():
        #     if info['type'] != 'variable' and info['type'] != 'combination':
        #         del new_library_info[name]
        # library_info = new_library_info
        self.human_message_to_combine_model = textwrap.dedent(
            f''' 
            You are respeonsible to form a new car following model, whose physical unit should be (m/s^2). 

            Here are symbols you can use to assist you to form new symbols, new model can only contain symbols in this dictionary:
            {delimiter} Symbol Library:
            {dict2block(library_info)}

            Your output should keep the same format as the following example:
            {delimiter} New car following model:
            {dict2block({'explaination': '...', 'prefix_expression': ['...'], 'infix_expression': '...', 'units': ['...']})}

            Then, please form only one new model based on the symbol library. You are not allowed to use any classical car-following model directly, such as IDM, Gipps, etc.
            When combining new model, you should start from the basic symbols, step by step combine more complex combinations, and consider why it can construct a car following model and whether the physical units meet the constraints. 
            
            {delimiter}Here is a example reanoning process you should follow:
            1. I choose symbols ... to combine new symbol ... because ..., the analytical process of the physical unit is ..., it's legal(or illegal, so I should combine again to meet the constraits of physical unit).
            ... (combine new symbols)
            N. Then I am about to output the new model with orginal symbols and new symbols I combined, the new model is ..., (why the new model can reflect the car-following behaviour), the physical unit is ..., it's legal(or illegal, so I should combine again to meet the constraits of physical unit).


            The car following model you output should reflect your reasoning process, and the model should be able to describe the car-following behavior well.
            You can stop reasoning once you make sure your new model can be used to well describe a car-following model. 
            To ensure decode your output to a python dictionary, "prefix_expression" should be a list containing the prefix expression of the new symbol, the type of elemnts in the list should be python str, 
            Before you give your final output, you should carefully check if the physical unit of the new model is not only (m/s^2)(for example, if your final model give as adding of items, you should check if every item's unit is m/s^2) but also legal(for example, you cannot add two symbols whose units are separately (m) and (s)).
            '''
        )
        self.human_message_to_combine_model = self.human_message_to_combine_model.replace("            ", "")

    def initialize_system_message_to_choose_symbol(self):
        self.system_message_to_choose_symbol = textwrap.dedent(
            f'''
            You are required to act as a helpful assistant, you can choose best symbol in a symbol library, which can be used to construct a complex car-following model.
            You should not only consider operator rules, but also unit constraints and physical meanings.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat until you decide to output your final answer>
            Response to user:
            {delimiter} <Your final output>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message_to_choose_symbol = self.system_message_to_choose_symbol.replace("            ", "")

    def initialize_human_message_to_choose_symbol(self, library_info):
        new_library_info = copy.deepcopy(library_info)
        for name, info in library_info.items():
            if info['type'] != 'variable' and info['type'] != 'combination':
                del new_library_info[name]
        library_info = new_library_info

        self.human_message_to_choose_symbol = textwrap.dedent(
            f'''
            Here are symbols which are used to construct car-following models.
            {delimiter} Symbol Library:
            {dict2block(library_info)}

            You are required to choose the best symbol in the symbol library, which can be used to construct a complex car-following model.
            You should not only consider operator rules, but also unit constraints and physical meanings of these symbols.

            Your final output should keep the same format as the following example:
            {delimiter} Best symbol:
            {dict2block({'name': '...', 'description': '...', 'units': ['...'], 'prefix_expression': ['...'], 'type': '...'})}

            When you are reasoning, you should compare your output symbol with other symbols which you think is competitive, and give your reason why you choose this symbol.
            Please note that complex symbols are not necessarily excellent. For complex symbols, you need to analyze whether their combination is reasonable and whether they can prefectly reflect some part of the car-following behavior.
            '''
        )
        self.human_message_to_choose_symbol = self.human_message_to_choose_symbol.replace("            ", "")

    def generate_prompt_to_combine_model(self, library_info):
        self.initialize_system_message_to_combine_model()
        self.initialize_human_message_to_combine_model(library_info)
        message = [
            SystemMessage(content=self.system_message_to_combine_model),
            HumanMessage(content=self.human_message_to_combine_model)
        ]

        for m in message:
            print(m.content)
            print("\n")

        return message

    def generate_prompt_to_choose_symbol(self, library_info):
        self.initialize_system_message_to_choose_symbol()
        self.initialize_human_message_to_choose_symbol(library_info)
        message = [
            SystemMessage(content=self.system_message_to_choose_symbol),
            HumanMessage(content=self.human_message_to_choose_symbol)
        ]

        for m in message:
            print(m.content)
            print("\n")

        return message
