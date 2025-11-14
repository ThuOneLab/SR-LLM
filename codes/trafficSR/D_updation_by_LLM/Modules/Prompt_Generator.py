import textwrap
from langchain.schema import HumanMessage, SystemMessage
from rich import print
import copy

from codes.trafficSR.D_updation_by_LLM.Modules.Agent_utils import *
from codes.trafficSR.D_updation_by_LLM.Modules.defaults import *

example_symbol_dict = {
    "prefix_expression": ["symbol 1", "symbol 2", "symbol 3", "..."],
    "infix_expression": "..."
}


class Prompt_Generator:
    def __init__(self,model_usage="car-following"):
        self.model_usage = model_usage
        self.initialize_system_message()
        self.initialize_rag_example_intro_message()
        
        self.initialize_physical_unit_message()
        self.initialize_check_legal_message()
        self.initialize_directly_build_message()
        
        self.initialize_find_in_expression_message()
        self.initialize_check_units_message()
        self.initialize_no_examples_message()
        self.initialize_cannot_directly_build_message()
        
        self.initialize_format_message()

    def initialize_system_message(self):
        self.system_message = textwrap.dedent(
            f'''
            You are required to act as a helpful assistant, you can provide guidance for constructing new symbols, making it easier for construct a complex {self.model_usage} model.
            Not only consider operator rules, but also unit constraints and physical meanings.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat until you decide to output your final answer>
            Response to user:
            {delimiter} <Your final output>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message = self.system_message.replace(interval, "")

    def generate_symbol_info_message(self, symbol_info: dict = DEFAULT_SYMBOL_INFO, library_info: dict = DEFAULT_LIBRARY_INFO):
        new_library_info = copy.deepcopy(library_info)
        # for name, info in library_info.items():
        #     del new_library_info[name]['prefix_expression']
        library_info = new_library_info
        variable_and_combination=[item for item in library_info.items() if item[1]['type']=='variable' or item[1]['type']=='combination']
        print("variable_and_combination:",variable_and_combination)
        self.info_message = textwrap.dedent(
            f'''
            Here is the current given symbol, new symbols must contain this symbol:
            {delimiter} Current given symbol: 
            {dict2block(symbol_info)}

            And here are current symbols you can use to assist you to form new symbols, new symbols can only contain symbols in this dictionary: 
            {delimiter} Symbol Library:
            {dict2block(library_info)}

            Then, please form only one new symbol based on the given symbol and the symbol library with a python dictionary format(must contain ```python ```).
            Your new symbol should keep the same format as the given symbol, and must contain the given symbol {symbol_info['name']}, and is not allowed to have same prefix expression as symbols in the symbol library.
            '''
        )
        self.info_message = self.info_message.replace(interval, "")

    def initialize_rag_example_intro_message(self):
        self.example_info_message = textwrap.dedent(
            f'''
            The messages above are some examples of how to construct a new symbol for constructing a {self.model_usage} model based on the given symbols.
            You can refer to those examples to form new symbols based on the following given symbols.
            '''
        )
        self.example_info_message = self.example_info_message.replace(interval, "")

    def generate_rag_example_message(self, fewshot_results):
        self.example_messages = []
        for i in range(len(fewshot_results)):
            example_message = f"{delimiter}Example {i + 1}:\n"
            example_message += f"Human question:\n{fewshot_results[i]['HumanMessage']}\n" + \
                               f"AI answer:\n{fewshot_results[i]['AIMessage']}\n" + \
                               f"AI reflection:\n{fewshot_results[i]['AIReflection']}\n"
            self.example_messages.append(example_message)

    def initialize_physical_unit_message(self):
        self.physical_unit_message = textwrap.dedent(
            f'''For better understanding, your new symbol's physical unit must only among this: 
            [1, 0](m, represent the distance item), [0, 1] (s, represent the time item), [1, -1] (m/s, represents the speed item), [1, -2] (m/s^2, represent the acceleration item), [0, 0] (dimensionless, represents the factor influence the car-following behavior), 
            [2, -2] (m^2/s^2, represents the relative change in speed between the ego vehicle and the preceding vehicle), [2, -4] (m^2/s^4, represents the overall dynamic response capability of the vehicle). 
            So, how can you combine the symbols in the symbol library to form a new symbol that has the same calculation formula as the given symbol. 
            The physical unit of the new symbol is among {[1, 0], [0, 1], [1, -1], [1, -2], [0, 0], [2, -2], [2, -4]}.
            Please check if your new symbol can meet this requirement, if not, you can revise it slightly, but not change the main structure of the symbol.
            '''
        )
        self.physical_unit_message = self.physical_unit_message.replace(interval, "")
    
    def initialize_check_legal_message(self):
        self.check_legal_message = textwrap.dedent(
            f'''You should reasoning if your new symbol is legal(for instance, you cannot add a speed to a distance due to mismatch of units). If illegal, you should correct your symbol.
            '''
        )
        self.check_legal_message = self.check_legal_message.replace(interval, "")
    
    def initialize_directly_build_message(self):
        # textwrap.dedent会自动检测并去除每行开头相同数量的空白字符，使得字符串内容左对齐，而不影响代码的缩进。
        self.directly_build_message = textwrap.dedent(
            f'''
            You should follow the following reasoning process step by step:
            (A-0) You can achieve the combination method of the example directly. 
            You need to prioritize combining the symbols in the symbol dictionary to form a new symbol that has same or similar combination method as the example above.
            You should firstly give out your new symbol which learns from the examples, then reason by the following steps (A-1)(A-2)(A-3):
            (A-1) Explain how you should use the current symbol and the symbols in the library to reproduce the combination of symbols in the examples.
            (A-2) {self.physical_unit_message}
            (A-3) {self.check_legal_message}
            '''
        )
        self.directly_build_message = self.directly_build_message.replace(interval, "")

    def generate_expression_message(self, symbol_info, expression_info):
        def dict2block(dict_obj, block_type='python'):
            dict_str = json.dumps(dict_obj, indent=4)
            markdown_str = f"```{block_type}\n{dict_str}\n```\n"
            return markdown_str

        self.expression_message = textwrap.dedent(
            f'''
            We also input a set of x-y data and fit some formula expressions for the car following algorithm based on these symbols.
            In the following expressions, they all contain the given symbol {symbol_info['name']}, so you can also refer to the combination method in the expression.
            Among them, rewards are all within the range of [0,1].
            The closer the reward is to 1, the better the performance of the formula, and the more attention needs to be paid to its combination method.
            '''
        )
        self.expression_message = self.expression_message.replace(interval, "")
        expression_md = dict2block(expression_info)
        self.expression_message += expression_md
        return expression_md

    def initialize_find_in_expression_message(self):
        self.find_in_expression_message = textwrap.dedent(
            f'''You should find the parts of these expressions that contain the given symbol and analyze what combination methods these expressions use to achieve excellent results.
            For example, if "alpha * v * * 2/v0 * * 2+alpha" often appears in excellent expressions, and the given symbol is "alpha*v**2/v0**2", then you can learn this combination of "alpha*v**2/v0**2+alpha".
            '''
        )
        self.find_in_expression_message = self.find_in_expression_message.replace(interval, "")
    
    def initialize_check_units_message(self):
        self.check_units_message = textwrap.dedent(
            f'''Finally, check if the physical unit of your symbol is correct, if not, you should generate another one.
            '''
        )
        self.check_units_message = self.check_units_message.replace(interval, "")
    
    def initialize_no_examples_message(self):
        self.no_examples_message = textwrap.dedent(
            f'''
            You should follow the following reasoning process step by step:
            (B-0) You should focus on the outstanding expressions provided, capture and learn the combination methods within them.
            You need to reason by the following steps (B-1)(B-2)(B-3)(B-4):
            (B-1) {self.find_in_expression_message}
            (B-2) {self.physical_unit_message}
            (B-3) {self.check_legal_message}
            (B-4) {self.check_units_message}
            '''
        )
        self.no_examples_message = self.no_examples_message.replace(interval, "")
    
    def initialize_cannot_directly_build_message(self):
        self.cannot_directly_build_message = textwrap.dedent(
            f'''
            You should follow the following reasoning process step by step:
            (B-0) Because you cannot achieve the examples directly, so you not only need to refer to the examples, but also need to refer to the expressions provided. 
            But you should pay more attention to the expressions provided and less attention to the examples provided.
            You need to reason by the following steps (B-1)(B-2)(B-3)(B-4):
            (B-1) {self.find_in_expression_message}
            If symbols with similar physical meanings to the examples can be selected from the library and combined using a combination method similar to the examples, then the new symbols would be even better.
            (B-2) {self.physical_unit_message}
            (B-3) {self.check_legal_message}
            (B-4) {self.check_units_message}
            '''
        )
        self.cannot_directly_build_message = self.cannot_directly_build_message.replace(interval, "")
    
    def initialize_symbol_name_message(self):
        self.symbol_name_message = textwrap.dedent(
            f'''
            Name requires the naming of symbols as 'x_y': 
            x reflects the unit of symbol('s_y' represnts [1, 0], 't_y' represents [0, 1], 'v_y' represents [1, -1],'a_y' represents [1, -2], 'factor_y' represents [0, 0], 'v2_y' representes [2, -2], 'a2_y' represents [2, -4]), and y represents the condense meaning of this symbol in the driving scenario(but not calculation method like x_v_div_t); 
            '''
        )
        self.symbol_name_message = self.symbol_name_message.replace(interval, "")
        
    def initialize_symbol_description_message(self):
        self.symbol_description_message = textwrap.dedent(
            f'''
            The description requires summarizing the physical meaning of a symbol in a simple sentence, so that readers can easily understand what the symbol represents, the description should follow the fromat like this:
            "The symbol in ...(distance/time/speed/acceleration/dimensionless/productSpeed/productAcceleration) unit which represents ...".
            '''
        )
        self.symbol_description_message = self.symbol_description_message.replace(interval, "")
    
    def initialize_format_message(self):
        self.initialize_symbol_name_message()
        self.initialize_symbol_description_message()
        self.format_message = textwrap.dedent(
            f'''
            You can stop reasoning once you make sure your new symbols is correct and legal, which can be used to construct model.
            Your final output should only contain a python dictionary of the new symbol.
            To ensure decode your output to a python dictionary, "prefix_expression" should be a list containing the prefix expression of the new symbol, the type of elemnts in the list should be python str.
            Please further simplify the content of "name" and "description" as below. 
            {self.symbol_name_message}
            {self.symbol_description_message}
            '''
        )
        self.format_message = self.format_message.replace(interval, "")

    def generate_prompt(self, high_score_symbols_info, library_info, fewshot_results, not_achieved, build_directly, expression_info=None, ):
        message = [
            SystemMessage(content=self.system_message),
        ]
        self.generate_rag_example_message(fewshot_results=fewshot_results) # 生成RAG知识例子的文本
        self.generate_symbol_info_message(high_score_symbols_info, library_info)
        useful_examples = []
        expression_md = ""
        if any(build_directly) == True:  # exist build directly
            for i in range(len(fewshot_results)):
                if build_directly[i]:
                    useful_examples.append(self.example_messages[i])
                    human_message = self.example_messages[i]
                    human_message += self.example_info_message
                    human_message += self.info_message
                    human_message += self.directly_build_message
                    human_message += self.format_message
                    message.append(HumanMessage(content=human_message))
        else:  # no exist build directly, we need to resort to RAG+DRL different expressions
            expression_md = self.generate_expression_message(high_score_symbols_info, expression_info)
            # only use examples that have not been created
            useful_example = ""
            for i in range(len(fewshot_results)):
                if not_achieved[i]:
                    useful_example += self.example_messages[i]
            useful_examples.append(useful_example)

            human_message = ""
            if useful_example:  # have examples not achieved
                human_message += useful_example
                human_message += self.example_info_message
            human_message += self.info_message
            human_message += self.expression_message
            if useful_example:  # have examples not achieved
                human_message += self.cannot_directly_build_message
            else:  # all examples achieved, then input no examples
                human_message += self.no_examples_message
            human_message += self.format_message
            message.append(HumanMessage(content=human_message))
        return useful_examples, expression_md, message

class Prompt_Generator_Feyn(Prompt_Generator):
    def __init__(self):
        super().__init__(model_usage="physical system")
        self.initialize_same_similar_to_example_message()
        self.initialize_directly_build_message_feyn()
        self.initialize_physical_unit_message_feyn()
        self.initialize_no_examples_message_feyn()
        self.initialize_cannot_directly_build_message_feyn()
        self.initialize_format_message_feyn()
    
    def initialize_same_similar_to_example_message(self):
        self.same_to_example_message = textwrap.dedent(
            f'''You can try to make use of the same symbol names, descriptions and combination method as examples, paying particular attention to their mid-fix expressions, so that you can choose the right operators to combine a new symbols.
            '''
        )
        self.same_to_example_message = self.same_to_example_message.replace(interval, "")
        
        self.similar_to_example_message = textwrap.dedent(
            f'''You can try to make use of the similar symbols and combination methods which are shown in examples, paying particular attention to mid-fix expressions of combination methods, so that you can choose the right operators to combine a new symbols.
            If symbols with similar physical meanings to the examples can be selected from the library and combined using a combination method similar to the examples, then the new symbols would be even better.
            '''
        )
        self.similar_to_example_message = self.similar_to_example_message.replace(interval, "")
    
    def initialize_directly_build_message_feyn(self):
        # textwrap.dedent会自动检测并去除每行开头相同数量的空白字符，使得字符串内容左对齐，而不影响代码的缩进。
        self.directly_build_message = textwrap.dedent(
            f'''
            You should follow the following reasoning process step by step:
            (A-0) You can achieve the combination method of the example directly. 
            You need to prioritize combining the symbols in the symbol dictionary to form a new symbol that has same or similar combination method as the example above.
            You should firstly give out your new symbol which learns from the examples, then reason by the following steps (A-1)(A-2)(A-3):
            (A-1) Explain how you should use the current symbol and the symbols in the library to reproduce the combination of symbols in the examples.
            (A-2) {self.same_to_example_message}
            (A-3) {self.check_legal_message}
            '''
        )
        self.directly_build_message = self.directly_build_message.replace(interval, "")
        
    def initialize_physical_unit_message_feyn(self):
        self.physical_unit_message = textwrap.dedent(
            f'''For better understanding, your new symbol's physical unit has to be a relatively common type; types that are too rare may not be used very often.
            Please check if your new symbol can meet this requirement, if not, you can revise it slightly, but not change the main structure of the symbol.
            '''
        )
        self.physical_unit_message = self.physical_unit_message.replace(interval, "")
    
    def initialize_no_examples_message_feyn(self):
        self.no_examples_message = textwrap.dedent(
            f'''
            You should follow the following reasoning process step by step:
            (B-0) You should focus on the outstanding expressions provided, capture and learn the combination methods within them.
            You need to reason by the following steps (B-1)(B-2)(B-3):
            (B-1) {self.find_in_expression_message}
            (B-2) {self.check_legal_message}
            (B-3) {self.physical_unit_message}
            '''
        )
        self.no_examples_message = self.no_examples_message.replace(interval, "")
    
    def initialize_cannot_directly_build_message_feyn(self):
        self.cannot_directly_build_message = textwrap.dedent(
            f'''
            You should follow the following reasoning process step by step:
            (B-0) Because you cannot achieve the examples directly, so you not only need to refer to the examples, but also need to refer to the expressions provided. 
            But you should pay more attention to the expressions provided and less attention to the examples provided.
            You need to reason by the following steps (B-1)(B-2)(B-3):
            (B-1) {self.find_in_expression_message}
            (B-2) {self.similar_to_example_message}
            (B-3) {self.check_legal_message}
            (B-4) {self.physical_unit_message}
            '''
        )
        self.cannot_directly_build_message = self.cannot_directly_build_message.replace(interval, "")
    
    def initialize_symbol_description_message_feyn(self):
        self.symbol_description_message = textwrap.dedent(
            f'''
            The description requires summarizing the physical meaning of a symbol in a simple sentence, so that readers can easily understand what the symbol represents, the description should follow the fromat like this:
            "The symbol which represents ...".
            '''
        ) # The symbol in ...(distance/time/speed/acceleration/dimensionless/productSpeed/productAcceleration) unit which represents ...
        self.symbol_description_message = self.symbol_description_message.replace(interval, "")
    
    def initialize_format_message_feyn(self):
        self.initialize_symbol_description_message_feyn()
        self.format_message = textwrap.dedent(
            f'''
            You can stop reasoning once you make sure your new symbols is correct and legal, which can be used to construct model.
            Your final output should only contain a python dictionary of the new symbol.
            To ensure decode your output to a python dictionary, "prefix_expression" should be a list containing the prefix expression of the new symbol, the type of elemnts in the list should be python str.
            Please further simplify the content of "description" as below. 
            {self.symbol_description_message}
            '''
        )
        self.format_message = self.format_message.replace(interval, "")

class Prompt_Generator2024:
    def __init__(self):
        pass

    def initialize_system_message(self):
        self.system_message = textwrap.dedent(
            f'''
            You are required to act as a helpful assistant, you can provide guidance for constructing new symbols, making it easier for construct a complex car-following model.
            Not only consider operator rules, but also unit constraints and physical meanings.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat until you decide to output your final answer>
            Response to user:
            {delimiter} <Your final output>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message = self.system_message.replace(interval, "")

    def initialize_human_message(self, symbol_info: dict = DEFAULT_SYMBOL_INFO,
                                 library_info: dict = DEFAULT_LIBRARY_INFO):
        new_library_info = copy.deepcopy(library_info)
        # for name, info in library_info.items():
        #     del new_library_info[name]['prefix_expression']
        library_info = new_library_info
        self.human_message = textwrap.dedent(
            f'''
            The messages above are some examples of how to construct a new symbol for constructing a car-following model based on the given symbols(However, perhaps there has no examples too). 
            The symbols you are about to use have similar physical meanings to the symbols used to form new symbols in these examples. 
            You should refer to those examples to form new symbols based on the current given symbols. You must use similar computational structures.
            
            Here is the current given symbol, new symbols must contain this symbol:
            {delimiter} Current given symbol: 
            {dict2block(symbol_info)}

            And here are symbols you can use to assist you to form new symbols, new symbols can only contain symbols in this dictionary:
            {delimiter} Symbol Library:
            {dict2block(library_info)}

            Then, please form only one new symbol based on the given symbol and the symbol library with a python dictionary format(must contain ```python ```). Your new symbol should keep the same format as the given symbol, and must contain the given symbol {symbol_info['name']}, and is not allowed to have same prefix expression as symbols in the symbol library.
            
            You should follow the following reasoning process step by step:
            (First of all) Analyze if combination methods of some examples have been achieved by current symbol. you  should ignore the examples which has been achieved completely by the current symbol or library because this kind of examples are not helpful for you to form new symbols.
            (Secondly) Analyze if the combination method of the examples can achieve directly by the current given symbol and dictionary, 
            which means that you can fully use the combination method in the example to combine new symbols, although some symbols are not in the libary, but you can find an alternative one which has similar physical meaning. 
            Please note that although the current symbol may not be exactly the same as the used symbols in the example, if their physical units are consistent, you can still try combining new characters using the combination method in the example.
            Then you should choose whether to use A or B to reason according to whether you can generate a new symbol according to the example,
            If you can achieve the combination method of the examples directly, you should follow the following steps (A-0)(A-1)(A-2)(A-3); else, you should follow the following steps (B-0)(B-1)(B-2)(B-3)(B-4).

            (A-0)If you can achieve the combination method of the examples directly, you need to prioritize combining the symbols in the symbol dictionary to form a new symbol that has same or similar combination method as examples above.
            You should firstly give out your new symbol which learns from the examples, then reason by the following steps (A-1)(A-2)(A-3):
            (A-1)Which example you think current symbol can achieve, give out the content of this example(which means you should give out the operation details in the example), and explain how you should use the current symbol and the symbols in the library to reproduce the combination of symbols in the examples.
            (A-2)For better understanding, your new symbol's physical unit must only among this: [1,0](m, represent the distance item), [0, 0] (dimensionless, represents the factor influence the car-following behavior), [0, 1] (s, represent the time item), [1,-1] (m/s, represnt the speed item),s [1, -2] (m/s^2, represent the acceleration item). So, how can you combine the symbols in the symbol library to form a new symbol that has the same calculation formula as the given symbol, and the physical unit of the new symbol is among {[1, 0], [0, 0], [0, 1], [1, -2]}.
            Please check if your new symbol can meet this requirement, if not, you can revise it slightly, but not change the main structure of the symbol.
            (A-3)You should reasoning if your new symbol is legal(for instance, you cannot add a speed to a distance). If illegal, you should correct your symbol.
            

            (B-0)If you cannot achieve examples or there are no examples, then reason by the following steps (B-1)(B-2)(B-3)(B-4):
            (B-1) For better understanding, your new symbol's physical unit must only among this: [1,0](m, represent the distance item), [0, 0] (dimensionless, represents the factor influence the car-following behavior), [0, 1] (s, represent the time item), [1,-1] (m/s, represnt the speed item),s [1, -2] (m/s^2, represent the acceleration item). So, how can you combine the symbols in the symbol library to form a new symbol that has the same calculation formula as the given symbol, and the physical unit of the new symbol is among {[1, 0], [0, 0], [0, 1], [1, -2]}.
            (B-2) You should reasoning how to combine a good symbol to reflect some factors in car-following model, which physical units must fit constraits in (2), which means the physical unit of the new symbol must be among {[1, 0], [0, 0], [0, 1], [1, -1], [1, -2]}.
            (B-3) You should reasoning if your new symbol is legal(for instance, you cannot add a speed to a distance). If illegal, you should generate another legal one.
            (B-4) Finally, check if the physical unit of your symbol is correct, if not, you should generate another one.

            
            
            You can stop reasoning once you make sure your new symbols can be used to construct a car-following model. Your final output should only contain a python dictionary of the new symbol.
            To ensure decode your output to a python dictionary, "prefix_expression" should be a list containing the prefix expression of the new symbol, the type of elemnts in the list should be python str.
            Please further simplify the content of "name" and "description". Name requires the naming of symbols as 'x_y', where x reflects the unit of symbol('s_y' represnts [1, 0], 't_y' represents [0, 1], 'a_y' represents [1, -2], 'factor_y' represents [0, 0]), and y represents the condense meaning of this symbol in the driving scenario(but not calculation method like x_v_div_t); 
            The description requires summarizing the physical meaning of a symbol in a simple sentence, so that readers can easily understand what the symbol represents, the description should follow the fromat like this:
            "The symbol in ...(distance/time/acceleration/dimensionless) units which represents ...".
            '''
        )
        # You should follow the following reasoning process step by step with (1)(2)(3)(4):
        # (1) Analyze if the combination method of the examples can achieve directly by the current given symbol and dictionary. 
        # If possible, you need to prioritize combining the symbols in the symbol dictionary to form a new symbol that has same or similar combination method as examlpes above.
        # After you give your new symbol which learns from the examples, you can directly jump to step (4) to check legitimacy of your new symbol.
        # Please note that although the current symbol may not be exactly the same as the used symbols in the example, if their physical units are consistent, you can still try combining new characters using the combination method in the example.
        # (If there are no examples, you can ignore this step; if there is more than one example, you can choose one you think better to follow.)
        # (2) For better understanding, your new symbol's physical unit must only among this: [1,0](m, represent the distance item), [0, 0] (dimensionless, represents the factor influence the car-following behavior), [0, 1] (s, represent the time item), [1,-1] (m/s, represnt the speed item),s [1, -2] (m/s^2, represent the acceleration item). So, how can you combine the symbols in the symbol library to form a new symbol that has the same calculation formula as the given symbol, and the physical unit of the new symbol is among {[1,0], [0,0], [0,1], [1,-2]}.
        # (3) You should reasoning how to combine a good symbol to reflect some factors in car-following model, which physical units must fit constraits in (2), which means the physical unit of the new symbol must be among {[1,0], [0,0], [0,1],[1,-1], [1,-2]}.
        # (4) You should reasoning if your new symbol is legal(for instance, you cannot add a speed to a distance). If illegal, you should generate another legal one.
        # (5) Finally, check if the physical unit of your symbol is correct, if not, you should generate another one.
        self.human_message = self.human_message.replace(interval, "")

    def initialize_human_message_totally_based_on_knowledge(self, symbol_info: dict = DEFAULT_SYMBOL_INFO,
                                                            library_info: dict = DEFAULT_LIBRARY_INFO):
        new_library_info = copy.deepcopy(library_info)
        # for name, info in library_info.items():
        #     del new_library_info[name]['prefix_expression']
        library_info = new_library_info
        self.human_message = textwrap.dedent(
            f'''
            The messages above are some examples of how to construct a new symbol for constructing a car-following model based on the given symbols(However, perhaps there has no examples too). 
            The symbols you are about to use have similar physical meanings to the symbols used to form new symbols in these examples. 
            You need to fully refer to those examples to form new symbols based on the current given symbols. The combination method of your new symbol must be among these examples.
            
            Here is the current given symbol, new symbols must contain this symbol:
            {delimiter} Current given symbol: 
            {dict2block(symbol_info)}

            And here are symbols you can use to assist you to form new symbols, new symbols can conly contain symbols in this dictionary:
            {delimiter} Symbol Library:
            {dict2block(library_info)}

            Then, please form only one new symbol based on the given symbol and the symbol library with a python dictionary format(must contain ```python ```). Your new symbol should keep the same format as the given symbol, and must contain the given symbol {symbol_info['name']}, and is not allowed to have same prefix expression as symbols in the symbol library.
            
            You should follow the following reasoning process step by step:
            (First of all) Analyze all the examples, give out the prefix expression(prefix expression is a mathematical notation where the operator precedes the operands, such as in "plus 3 4" to represent the addition of 3 and 4.) of the combined symbols in the examples with python block in the following format: 
            {dict2block(example_symbol_dict)}

            (Secondly) Analyze each example, reason which example's prefix expression contains the current symbol {symbol_info['name']}(or have similar physical meaning), and its prefix expression should not appear in the symbol libary either.
            Choose one example that contains the current symbol {symbol_info['name']} and its prefix expression should not appear in the symbol libary either.
            If there is no example which not only contains the current symbol {symbol_info['name']} but also its prefix expression donot appear in the symbol libary either, you can directly point out that there is no example fit the requirements, then give out your final output.
            
            (Thirdly) Combine new symbol based on 1.current symbol, 2.symbol library and 3.the example you choose. You should combine the current symbol with symbol library, to get a new symbol completedly same as the example you choose.

            
            Explain how you should use the current symbol and the symbols in the library to reproduce the combination of symbols in the examples.
            You can stop reasoning once you make sure your new symbols can be used to construct a car-following model. Your final output should only contain a python dictionary of the new symbol.
            To ensure decode your output to a python dictionary, "prefix_expression" should be a list containing the prefix expression of the new symbol, the type of elemnts in the list should be python str.
            Please further simplify the content of "name" and "description". Name requires the naming of symbols as 'x_y', where x reflects the unit of symbol('s_y' represnts [1, 0], 't_y' represents [0, 1], 'a_y' represents [1, -2], 'factor_y' represents [0, 0]), and y represents the condense meaning of this symbol in the driving scenario(but not calculation method like x_v_div_t); 
            The description requires summarizing the physical meaning of a symbol in a simple sentence, so that readers can easily understand what the symbol represents, the description should follow the fromat like this:
            "The symbol in ...(distance/time/acceleration/dimensionless) units which represents ...".

            '''
        )
        # You should follow the following reasoning process step by step with (1)(2)(3)(4):
        # (1) Analyze if the combination method of the examples can achieve directly by the current given symbol and dictionary. 
        # If possible, you need to prioritize combining the symbols in the symbol dictionary to form a new symbol that has same or similar combination method as examlpes above.
        # After you give your new symbol which learns from the examples, you can directly jump to step (4) to check legitimacy of your new symbol.
        # Please note that although the current symbol may not be exactly the same as the used symbols in the example, if their physical units are consistent, you can still try combining new characters using the combination method in the example.
        # (If there are no examples, you can ignore this step; if there is more than one example, you can choose one you think better to follow.)
        # (2) For better understanding, your new symbol's physical unit must only among this: [1,0](m, represent the distance item), [0, 0] (dimensionless, represents the factor influence the car-following behavior), [0, 1] (s, represent the time item), [1,-1] (m/s, represnt the speed item),s [1, -2] (m/s^2, represent the acceleration item). So, how can you combine the symbols in the symbol library to form a new symbol that has the same calculation formula as the given symbol, and the physical unit of the new symbol is among {[1,0], [0,0], [0,1], [1,-2]}.
        # (3) You should reasoning how to combine a good symbol to reflect some factors in car-following model, which physical units must fit constraits in (2), which means the physical unit of the new symbol must be among {[1,0], [0,0], [0,1],[1,-1], [1,-2]}.
        # (4) You should reasoning if your new symbol is legal(for instance, you cannot add a speed to a distance). If illegal, you should generate another legal one.
        # (5) Finally, check if the physical unit of your symbol is correct, if not, you should generate another one.
        self.human_message = self.human_message.replace(interval, "")

    def generate_prompt(self, high_score_symbols_info, library_info, fewshot_results):
        self.initialize_system_message()
        self.initialize_human_message(high_score_symbols_info, library_info)
        message = [
            SystemMessage(content=self.system_message),
        ]

        example_message = f"{delimiter}Examples:\n"
        for i in range(len(fewshot_results)):
            example_message += f"Human question:\n{fewshot_results[i]['HumanMessage']}\nAI answer:\n{fewshot_results[i]['AIMessage']}\n"
        self.human_message = example_message + self.human_message
        message.append(HumanMessage(content=self.human_message))

        for m in message:
            print(m.content)
            print("\n")

        return message

    def generate_prompt_totally_based_on_knowledge(self, high_score_symbols_info, library_info, fewshot_results):
        self.initialize_system_message()
        self.initialize_human_message_totally_based_on_knowledge(high_score_symbols_info, library_info)
        message = [
            SystemMessage(content=self.system_message),
        ]

        example_message = f"{delimiter}Examples:\n"
        for i in range(len(fewshot_results)):
            example_message += f"Human question:\n{fewshot_results[i]['HumanMessage']}\nAI answer:\n{fewshot_results[i]['AIMessage']}\n"
        self.human_message = example_message + self.human_message
        message.append(HumanMessage(content=self.human_message))

        for m in message:
            print(m.content)
            print("\n")

        return message

class Prompt_Generator_Random(Prompt_Generator):
    def __init__(self):
        super().__init__(model_usage="general symbolic regression")
        self.initialize_system_message_random()
        self.initialize_same_similar_to_example_message()
        self.initialize_directly_build_message_random()
        self.initialize_physical_unit_message_random()
        self.initialize_no_examples_message_random()
        self.initialize_cannot_directly_build_message_random()
        self.initialize_format_message_random()

    def initialize_system_message_random(self):
        self.system_message = textwrap.dedent(
            f'''
            You are required to act as a helpful assistant, you can provide guidance for constructing new symbols, making it easier for construct a complex {self.model_usage} model.
            Not only consider operator rules, but also unit constraints and physical meanings.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat until you decide to output your final answer>
            Response to user:
            {delimiter} <Your final output>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message = self.system_message.replace(interval, "")

    def initialize_same_similar_to_example_message(self):
        self.same_to_example_message = textwrap.dedent(
            f'''You can try to make use of the same symbol names, descriptions and combination method as examples, paying particular attention to their mid-fix expressions, so that you can choose the right operators to combine a new symbols.
            '''
        )
        self.same_to_example_message = self.same_to_example_message.replace(interval, "")
        
        self.similar_to_example_message = textwrap.dedent(
            f'''You can try to make use of the similar symbols and combination methods which are shown in examples, paying particular attention to mid-fix expressions of combination methods, so that you can choose the right operators to combine a new symbols.
            If symbols with similar meanings to the examples can be selected from the library and combined using a combination method similar to the examples, then the new symbols would be even better.
            '''
        )
        self.similar_to_example_message = self.similar_to_example_message.replace(interval, "")
    
    def initialize_directly_build_message_random(self):
        # textwrap.dedent会自动检测并去除每行开头相同数量的空白字符，使得字符串内容左对齐，而不影响代码的缩进。
        self.directly_build_message = textwrap.dedent(
            f'''
            You should follow the following reasoning process step by step:
            (A-0) You can achieve the combination method of the example directly. 
            You need to prioritize combining the symbols in the symbol dictionary to form a new symbol that has same or similar combination method as the example above.
            You should firstly give out your new symbol which learns from the examples, then reason by the following steps (A-1)(A-2)(A-3):
            (A-1) Explain how you should use the current symbol and the symbols in the library to reproduce the combination of symbols in the examples.
            (A-2) {self.same_to_example_message}
            (A-3) {self.check_legal_message}
            '''
        )
        self.directly_build_message = self.directly_build_message.replace(interval, "")
        
    def initialize_physical_unit_message_random(self):
        self.physical_unit_message = textwrap.dedent(
            f'''For better understanding, your new symbol's physical unit has to be a relatively common type; types that are too rare may not be used very often.
            Please check if your new symbol can meet this requirement, if not, you can revise it slightly, but not change the main structure of the symbol.
            '''
        )
        self.physical_unit_message = self.physical_unit_message.replace(interval, "")
    
    def initialize_no_examples_message_random(self):
        self.no_examples_message = textwrap.dedent(
            f'''
            You should follow the following reasoning process step by step:
            (B-0) You should focus on the outstanding expressions provided, capture and learn the combination methods within them.
            You need to reason by the following steps (B-1):
            (B-1) {self.find_in_expression_message}
            '''
        )
        self.no_examples_message = self.no_examples_message.replace(interval, "")
    
    def initialize_cannot_directly_build_message_random(self):
        self.cannot_directly_build_message = textwrap.dedent(
            f'''
            You should follow the following reasoning process step by step:
            (B-0) Because you cannot achieve the examples directly, so you not only need to refer to the examples, but also need to refer to the expressions provided. 
            But you should pay more attention to the expressions provided and less attention to the examples provided.
            You need to reason by the following steps (B-1)(B-2)(B-3):
            (B-1) {self.find_in_expression_message}
            (B-2) {self.similar_to_example_message}
            (B-3) {self.check_legal_message}
            '''
        )
        self.cannot_directly_build_message = self.cannot_directly_build_message.replace(interval, "")
    
    def initialize_symbol_description_message_random(self):
        self.symbol_description_message = textwrap.dedent(
            f'''
            The description requires summarizing the physical meaning of a symbol in a simple sentence, so that readers can easily understand what the symbol represents, the description should follow the fromat like this:
            "The symbol which represents ...".
            '''
        ) # The symbol in ...(distance/time/speed/acceleration/dimensionless/productSpeed/productAcceleration) unit which represents ...
        self.symbol_description_message = self.symbol_description_message.replace(interval, "")
    
    def initialize_format_message_random(self):
        self.initialize_symbol_description_message_random()
        self.format_message = textwrap.dedent(
            f'''
            You can stop reasoning once you make sure your new symbols is correct and legal, which can be used to construct model.
            Your final output should only contain a python dictionary of the new symbol.
            To ensure decode your output to a python dictionary, "prefix_expression" should be a list containing the prefix expression of the new symbol, the type of elemnts in the list should be python str.
            Please further simplify the content of "description" as below. 
            {self.symbol_description_message}
            '''
        )
        self.format_message = self.format_message.replace(interval, "")


if __name__ == "__main__":
    pg = Prompt_Generator()
    print(pg.generate_prompt(high_score_symbols_info=DEFAULT_SYMBOL_INFO, library_info=DEFAULT_LIBRARY_INFO, fewshot_results=[]))
