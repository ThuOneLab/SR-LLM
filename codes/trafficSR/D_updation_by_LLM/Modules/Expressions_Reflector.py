from langchain.schema import AIMessage, SystemMessage
import textwrap
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_utils import *

delimiter = "####"
example_knowledge = {
    "source": "LLM",
    "content": "LLM agent suggests that the difference between the following distance $s$ and the safe following distance $s_{safe}$ can be used to form a distance influence factor to describe the impact of distance factors on the final following behavior. The calculation of the distance influence factor then uses the following distance $s$ and the safe following distance $s_{safe}$. The calculation method is to first divide the safe following distance from the true following distance and then square it to further expand the influence of the symbol to obtain the combined symbol $(s_{safe}/s)^2$.",
}

example_dictionary = {
    "infix_expression": "...",
    "prefix_expression": ["...(list of symbols)"],
    "units": ["...(represents meter)", '...(represents second)'],
    "name": "...(str:x_y, where x reflects the unit of symbol('s_y' represnts [1, 0], 't_y' represents [0, 1], 'a_y' represents [1, -2], 'factor_y' represents [0, 0]), and y represents the condense meaning of this symbol in the driving scenario)",
    "is_final_expression": "...(bool:True or False)"
}
IDM_example_knowledge = {
            "source": "IDM",
            "formula":"alpha*(1-(v/v_0)^2-((s_0+T*v)+v*delta_v/(2*sqrt(alpha*b))/s)^2)",
            "key": '$v$: The symbol in speed unit which represents ego vehicle speed, ' + \
               '$v_0$: The symbol in speed unit which represents desired ego vehicle speed',
            "target": '$factor_v_ratio$: The symbol in dimensionless units which represents the ratio of current speed to desired speed',
            "content": "In order to reflect the proportional relationship between the speed of the ego vehicle $v$ and the desired speed of ego vehicle $v_0$, " + \
                   "human experts use $v$ as the dividend and $v_0$ as the divisor to a new symbol #v/v_0#. " + \
                   "When $v$ is greater than $v_0$, this term will increase. " + \
                   "When $v$ is less than $v_0$, this term will decrease. " + \
                   "It can be further condensed as the influencing factor of speed, marked as $factor_v_ratio$.",
            "comment": "Good symbol that reflects the proportional relationship between the current speed and the desired speed of ego vehicle. This operation also has excellent fitting performance.",
            "reflection": "It's a good symbol. I need to consider the current speed and the desired speed of ego vehicle."
        }

class Expressions_Reflector:
    def __init__(self):
        self.initialize_system_message_for_expression_reflector()

    def initialize_system_message_for_expression_reflector(self):
        self.system_message = textwrap.dedent(
            f'''
            You are a reflection assistant who is responsible for reflect the expression output from another reinforce learning agent.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message = self.system_message.replace("            ", "")

    def reflect_expression(self, expression_dictionary, symbol_library):
        symbol_used_list = list(set(expression_dictionary['prefix_expression']))
        symbol_used_dict = {}
        for symbol in symbol_used_list:
            info_dict = {}
            info_dict["name"] = symbol_library[symbol]["name"]
            info_dict["units"] = symbol_library[symbol]["units"]
            if symbol_library[symbol]["type"] != "free_const":
                info_dict["description"] = symbol_library[symbol]["description"]
            else:
                info_dict["description"] = "a free const symbol."
            symbol_used_dict[symbol] = info_dict

        reflect_message = textwrap.dedent(
            f"""
            You should give a reflection on the expression you received, then finally output some pieces of refined knowledge. The expression you received can fit the car following trajectory well, which means it can  reflect the car-following behavior.
            You should explain why it can be well used to combine the car following model, based on the physical meaning of the basic symbol used in this symbol and the combination structure of this symbol.

            {delimiter}Here is the symbol information of symbols in this symbol dictionary:
            {dict2block(symbol_used_dict)}

            {delimiter}Here is the expression information of the expression you received:
            {dict2block(expression_dictionary)}

            Your should follow the following thinking process one by one: 
            1. What is the combination method of this symbol? You should give out its detailed operation method.
            2. How can you hierarchy this expression and break it down into multiple more easily understandable units? Specifically, you can decompose the expression layer by layer from bottom to top, combining some symbols together and treating them as a whole symbol to simplify the expression. 
            Give out your detailed decomposition process. For better understanding, when you combing some symbols together, your new symbol's physical unit must only among this: 
            [1,0](m, represent the distance item), [0, 0] (dimensionless, represents the factor influence the car-following behavior), [0, 1] (s, represent the time item), 
            [1,-1] (m/s, represnt the speed item),s [1, -2] (m/s^2, represent the acceleration item). 
            3. Finally, you should summary your reasoning process, and give out a pieces of python dictionary like this(You should output aone piece of combination with 'is_final_expression' is True, and when your combination use other combination as a element in its infix and prefix expression, you should use the overall name of the element combination, if the expression is too simple, you can just give out one piece of combination with 'is_final_expression' is True):

            {delimiter} Combination k(where k represents the number of the combination, starting from 1)
            {dict2block(example_dictionary)} 
            (...)
            
            Finally, you can output your refined knowledges for each python dictionary you give out.
            Your should follow the following thinking process one by one to give out your refined knowledge(your refined knowledge should reflect these contents): 
            1. What is the combination method?  You should give out its infix expression and explain the operation of it.
            2. what is the physical meaning of the combination? Please note that there maybe free const symbols in the symbol dictionary you received, please check carefully and give a reasonable expaination to them.
            3. What is the physical meaning of this combination? 
            4. How can this physical meaning be further described as more concise and in line with the car-following scenario?
            
            {delimiter}Here is a example of knowledge, you must keep its output format and give your final output with a python dict.
            
            {delimiter} Knowledge k(where k represents the number of the knowledge, starting from 1, the number of the knowledge should be the same as the number of the combination you give out)
            {dict2block(example_knowledge)} 
            (...)

            You can stop reasoning once you know how to correct these errors. Your final output should follow the thinking step above, with format "Final reflection: <reflection...>".
            """
        )
        reflect_message = reflect_message.replace("            ", "")
        message = [
            SystemMessage(content=self.system_message),
            AIMessage(content=reflect_message),
        ]
        return message

class Expressions_Reflector_Feynman:
    def __init__(self):
        self.initialize_system_message_for_expression_reflector()


    def initialize_system_message_for_expression_reflector(self):
        self.system_message = textwrap.dedent(
            f'''
            You are a reflection assistant who is responsible for reflect the physical formula.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message = self.system_message.replace("            ", "")

    
    def reflect_expression(self, expression_dictionary):

        example_dictionary = {
            "infix_expression": "...",
            "name":"...(represents the condense meaning of this symbol in the driving scenario)",
            "is_final_expression": "...(bool:True or False)",
            "used_symbols": ["...(list of symbols used in this combination)"]
        }

        
        reflect_message = textwrap.dedent(
            f"""
            You should give a reflection on the expression you received, then finally output some pieces of refined knowledge.
            You should explain how the expression is organized and what is the physical meaning of the expression, 
            based on the physical meaning of the basic symbol used in this symbol and the combination structure of this symbol.


            {delimiter}Here is the expression information of the expression you received:
            {dict2block(expression_dictionary)}

            Your should follow the following thinking process one by one: 
            1. What is the combination method of this symbol? You should give out its detailed operation method.
            2. How can you hierarchy this expression and break it down into multiple more easily understandable units? Specifically, you can decompose the expression layer by layer from bottom to top, combining some symbols together and treating them as a whole symbol to simplify the expression. 
            Give out your detailed decomposition process. 
            3. Finally, you should summary your reasoning process, and give out a pieces of python dictionary like this(You should output aone piece of combination with 'is_final_expression' is True, and when your combination use other combination as a element in its infix and prefix expression, 
            you should use the overall name of the element combination, if the expression is too simple, you can just give out one piece of combination with 'is_final_expression' is True):

            {delimiter} Combination k(where k represents the number of the combination, starting from 1)
            {dict2block(example_dictionary)} 
            (...)
            
            Finally, you can output your refined knowledges for each python dictionary you give out.
            Your should follow the following thinking process one by one to give out your refined knowledge(your refined knowledge should reflect these contents): 
            1. What is the combination method?  You should give out its infix expression and explain the operation of it.
            2. what is the physical meaning of the combination? Please note that there maybe free const symbols in the symbol dictionary you received, please check carefully and give a reasonable expaination to them.
            3. What is the physical meaning of this combination? 
            4. How can this physical meaning be further described more concisely?
            
            {delimiter}Here is a example of knowledge, you must keep its output format and give your final output with a python dict.
            
            {delimiter} Knowledge k(where k represents the number of the knowledge, starting from 1, the number of the knowledge should be the same as the number of the combination you give out)
            {dict2block(example_knowledge)} 
            (...)

            You can stop reasoning once you know how to correct these errors. Your final output should follow the thinking step above, with format "Final reflection: <reflection...>".
            """
        )
        reflect_message = reflect_message.replace("            ", "")
        message = [
            SystemMessage(content=self.system_message),
            AIMessage(content=reflect_message),
        ]
        return message
    
    def reflect_expression2knowledge(self, expression_dictionary):
        # 运算符符号库M共有["add", "mul","sub", "div", “n4”, “n3”, "n2", "sqrt", "sin", "cos", "exp","log","arcsin"]，分别代表加、减、乘、除、四次方、三次方、平方、开方、sin、cos、指数函数、对数函数、反正弦函数。
        combination_dictionary = {
            "name":"...(represents the condense meaning of this symbol in the driving scenario)",
            "infix_expression": "...",
            "prefix_expression": ["...(list of symbols used in this combination)"],
            "physical_meaning": "The symbol which represents ...",
            "is_final_expression": "...(bool:True or False)",
        }
        
        example_combination = {
            "name": "squared_theta",
            "infix_expression": "theta**2",
            "prefix_expression": ["n2", "theta"],
            "is_final_expression": False,
        }
        
        knowledge_dictionary = {
            "source":f"...(based on the formula name of expression, i.e. {expression_dictionary['EquationName']})",
            "formula":f"...(based on the formula of expression, i.e. {expression_dictionary['Formula']})",
            "key": "$used_symbol_1$: the description of symbol 1, $used_symbol_2$: the description of symbol 2, ...(list of symbols that used in combination k. But it does not contain operators, only contains variables or constants. So the number of used_symbol depends on the number of variables and constants required for this combination)",
            "target": "$combination_name$: combination physical_meaning (The symbol which represents ...). ",
            "content":"must include the physical meaning of the combination and the combination infix_expression in #...#. Example: In order to...human experts use $used_symbol_1$ and $used_symbol_2$ to obtain the combined symbol #used_symbol_1 operator used_symbol_2#, which is marked as $combination_name$.",
            "comment": "comment on the combination (Example: Good symbol that reflects ...)",
            "reflection": "the reflection of the combination (Example: I need to consider ...)"
        }
        
        example_knowledge = {
            "source": "I.6.2a",
            "formula":"exp(-theta**2/2)/sqrt(2*pi)",
            "key": '$theta$: The symbol in dimensionless units which represents the angle',
            "target": '$squared_theta$: The symbol in dimensionless units which represents the square of theta',
            "content": "In order to build a new symbol that reflects the square of the angle $theta$, human experts use the symbol $theta$ as the base and then square it to obtain the combined symbol #theta^2#, which is marked as $squared_theta$.",
            "comment": "Good symbol that reflects the square of the angle $theta$. This operation has excellent fitting performance.",
            "reflection": "I need to consider the square of the angle $theta$."
        }
        
        example_knowledge2 = {
            "source": "I.6.2a",
            "formula":"exp(-theta**2/2)/sqrt(2*pi)",
            "key": '$c_1$: The symbol in dimensionless units which represents the constant 2',
            "target": '$squared_theta$: The symbol in dimensionless units which represents the square of theta',
            "content": "In order to build a new symbol that reflects the square of the angle $theta$, human experts use the symbol $theta$ as the base and then square it to obtain the combined symbol #theta^2#, which is marked as $squared_theta$.",
            "comment": "Good symbol that reflects the square of the angle $theta$. This operation has excellent fitting performance.",
            "reflection": "I need to consider the square of the angle $theta$."
        }
        
        reflect_message = textwrap.dedent(
            f"""
            You should give a reflection on the expression you received, then finally output some pieces of refined knowledge.
            You should explain how the expression is organized and what is the physical meaning of the expression, 
            based on the physical meaning of the basic symbol used in this symbol and the combination structure of this symbol. 


            {delimiter}Here is the expression information of the expression you received. Pay particular attention to its prefix expressions and infix expressions:
            {dict2block(expression_dictionary)}

            The operator symbol library has a total of ['add', 'mul', 'sub', 'div', 'n4', 'n3', 'n2', 'sqrt', 'sin', 'cos', 'exp', 'log', 'arcsin"], which stand for addition, subtraction, multiplication, division, quadratic, cubic, square, open square, sin, cos, exponential function, logarithmic function, and inverse sine function, respectively.
            
            Your should follow the following thinking process one by one: 
            1. What is the combination method of this symbol? You should give out its detailed operation method. 
            2. How can you hierarchy this expression and break it down into multiple more easily understandable units? Specifically, you can decompose the expression layer by layer from bottom to top, combining some symbols together and treating them as a whole symbol to simplify the expression. 
            Give out your detailed decomposition process. 
            3. Finally, you should summary your reasoning process, and give out a pieces of python dictionary like this(You should output aone piece of combination with 'is_final_expression' is True, and when your combination use other combination as a element in its infix and prefix expression, 
            you should use the overall name of the element combination, if the expression is too simple, you can just give out one piece of combination with 'is_final_expression' is True):

            {delimiter} Combination k(where k represents the number of the combination, starting from 1)
            {dict2block(combination_dictionary)} 
            (...)
            
            {delimiter}Here is Combination 1 case of formula 'exp(-theta**2/2)/sqrt(2*pi)' that you can reference it to output in this format. This is the innermost combination, on top of which more combinations will be put together, and so on, until the final combination is put together to form the combination of the whole formula.
            {dict2block(example_combination)} 
            (...)
            
            Finally, you can output your refined knowledges for each python dictionary you give out.
            Your should follow the following thinking process one by one to give out your refined knowledge(your refined knowledge should reflect these contents): 
            1. What is the combination method?  You should give out its infix expression and explain the operation of it.
            2. what is the physical meaning of the combination? Please note that there maybe free const symbols in the symbol dictionary you received, please check carefully and give a reasonable expaination to them.
            3. What is the physical meaning of this combination? 
            4. How can this physical meaning be further described more concisely?
            
            {delimiter}Here is a example of knowledge, you must keep its output format and give your final output with a python dict. 
            
            {delimiter} Knowledge k(where k represents the number of the knowledge, starting from 1, the number of the knowledge should be the same as the number of the combination you give out)
            {dict2block(knowledge_dictionary)} 
            (...)
            
            {delimiter}Here is Knowledge 1 case of formula 'exp(-theta**2/2)/sqrt(2*pi)' that you can reference it to output in this format. It corresponds to Combination 1 above and is the knowledge abstracted from it.
            {dict2block(example_knowledge)} 
            (...)

            You can stop reasoning once you know how to correct these errors. Your final output should follow the thinking step above, with format "Final reflection: <reflection...>".
            """
        )
        reflect_message = reflect_message.replace("            ", "")
        message = [
            SystemMessage(content=self.system_message),
            AIMessage(content=reflect_message),
        ]
        return message
    
class Expressions_Reflector_Random:
    def __init__(self):
        self.initialize_system_message_for_expression_reflector()


    def initialize_system_message_for_expression_reflector(self):
        self.system_message = textwrap.dedent(
            f'''
            You are a reflection assistant who is responsible for reflect the formula.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message = self.system_message.replace("            ", "")

    
    def reflect_expression(self, expression_dictionary, symbol_library):
        
        symbol_used_list = list(set(expression_dictionary['prefix_expression']))
        symbol_used_dict = {}
        for symbol in symbol_used_list:
            info_dict = {}
            info_dict["name"] = symbol_library[symbol]["name"]
            info_dict["units"] = symbol_library[symbol]["units"]
            if symbol_library[symbol]["type"] != "free_const":
                info_dict["description"] = symbol_library[symbol]["description"]
            else:
                info_dict["description"] = "a free const symbol."
            symbol_used_dict[symbol] = info_dict

        example_dictionary = {
            "infix_expression": "...",
            "name":"...(represents the condense meaning of this symbol in the driving scenario)",
            "is_final_expression": "...(bool:True or False)",
            "used_symbols": ["...(list of symbols used in this combination)"],
            "prefix_expression": ["...(list of symbols)"],
        }

        
        reflect_message = textwrap.dedent(
            f"""
            You should give a reflection on the expression you received, then finally output some pieces of refined knowledge.
            You should explain how the expression is organized and what is meaning of the expression, 
            based on the physical meaning of the basic symbol used in this symbol and the combination structure of this symbol.

            {delimiter}Here is the symbol information of symbols in this symbol dictionary:
            {dict2block(symbol_used_dict)}

            {delimiter}Here is the expression information of the expression you received:
            {dict2block(expression_dictionary)}

            Your should follow the following thinking process one by one: 
            1. What is the combination method of this symbol? You should give out its detailed operation method.
            2. How can you hierarchy this expression and break it down into multiple more easily understandable units? Specifically, you can decompose the expression layer by layer from bottom to top, combining some symbols together and treating them as a whole symbol to simplify the expression. 
            Give out your detailed decomposition process. 
            3. Finally, you should summary your reasoning process, and give out a pieces of python dictionary like this(You should output aone piece of combination with 'is_final_expression' is True, and when your combination use other combination as a element in its infix and prefix expression, 
            you should use the overall name of the element combination, if the expression is too simple, you can just give out one piece of combination with 'is_final_expression' is True):

            {delimiter} Combination k(where k represents the number of the combination, starting from 1)
            {dict2block(example_dictionary)} 
            (...)
            
            Finally, you can output your refined knowledges for each python dictionary you give out.
            Your should follow the following thinking process one by one to give out your refined knowledge(your refined knowledge should reflect these contents): 
            1. What is the combination method?  You should give out its infix expression and explain the operation of it.
            2. what is the physical meaning of the combination? Please note that there maybe free const symbols in the symbol dictionary you received, please check carefully and give a reasonable expaination to them.
            3. What is the physical meaning of this combination? 
            4. How can this physical meaning be further described more concisely?
            
            {delimiter}Here is a example of knowledge, you must keep its output format and give your final output with a python dict.
            
            {delimiter} Knowledge k(where k represents the number of the knowledge, starting from 1, the number of the knowledge should be the same as the number of the combination you give out)
            {dict2block(IDM_example_knowledge)} 
            (...)

            You can stop reasoning once you know how to correct these errors. Your final output should follow the thinking step above, with format "Final reflection: <reflection...>".
            """
        )
        reflect_message = reflect_message.replace("            ", "")
        message = [
            SystemMessage(content=self.system_message),
            AIMessage(content=reflect_message),
        ]
        return message
    
    def reflect_expression2knowledge(self, expression_dictionary):
        # 运算符符号库M共有["add", "mul","sub", "div", “n4”, “n3”, "n2", "sqrt", "sin", "cos", "exp","log","arcsin"]，分别代表加、减、乘、除、四次方、三次方、平方、开方、sin、cos、指数函数、对数函数、反正弦函数。
        combination_dictionary = {
            "name":"...(represents the condense meaning of this symbol in the driving scenario)",
            "infix_expression": "...",
            "prefix_expression": ["...(list of symbols used in this combination)"],
            "physical_meaning": "The symbol which represents ...",
            "is_final_expression": "...(bool:True or False)",
        }
        
        example_combination = {
            "name": "squared_theta",
            "infix_expression": "theta**2",
            "prefix_expression": ["n2", "theta"],
            "is_final_expression": False,
        }
        
        knowledge_dictionary = {
            "source":f"...(based on the formula name of expression, i.e. {expression_dictionary['EquationName']})",
            "formula":f"...(based on the formula of expression, i.e. {expression_dictionary['Formula']})",
            "key": "$used_symbol_1$: the description of symbol 1, $used_symbol_2$: the description of symbol 2, ...(list of symbols that used in combination k. But it does not contain operators, only contains variables or constants. So the number of used_symbol depends on the number of variables and constants required for this combination)",
            "target": "$combination_name$: combination physical_meaning (The symbol which represents ...). ",
            "content":"must include the physical meaning of the combination and the combination infix_expression in #...#. Example: In order to...human experts use $used_symbol_1$ and $used_symbol_2$ to obtain the combined symbol #used_symbol_1 operator used_symbol_2#, which is marked as $combination_name$.",
            "comment": "comment on the combination (Example: Good symbol that reflects ...)",
            "reflection": "the reflection of the combination (Example: I need to consider ...)"
        }
        
        example_knowledge = {
            "source": "I.6.2a",
            "formula":"exp(-theta**2/2)/sqrt(2*pi)",
            "key": '$theta$: The symbol in dimensionless units which represents the angle',
            "target": '$squared_theta$: The symbol in dimensionless units which represents the square of theta',
            "content": "In order to build a new symbol that reflects the square of the angle $theta$, human experts use the symbol $theta$ as the base and then square it to obtain the combined symbol #theta^2#, which is marked as $squared_theta$.",
            "comment": "Good symbol that reflects the square of the angle $theta$. This operation has excellent fitting performance.",
            "reflection": "I need to consider the square of the angle $theta$."
        }
        
        example_knowledge2 = {
            "source": "I.6.2a",
            "formula":"exp(-theta**2/2)/sqrt(2*pi)",
            "key": '$c_1$: The symbol in dimensionless units which represents the constant 2',
            "target": '$squared_theta$: The symbol in dimensionless units which represents the square of theta',
            "content": "In order to build a new symbol that reflects the square of the angle $theta$, human experts use the symbol $theta$ as the base and then square it to obtain the combined symbol #theta^2#, which is marked as $squared_theta$.",
            "comment": "Good symbol that reflects the square of the angle $theta$. This operation has excellent fitting performance.",
            "reflection": "I need to consider the square of the angle $theta$."
        }
        
        reflect_message = textwrap.dedent(
            f"""
            You should give a reflection on the expression you received, then finally output some pieces of refined knowledge.
            You should explain how the expression is organized and what is the physical meaning of the expression, 
            based on the physical meaning of the basic symbol used in this symbol and the combination structure of this symbol. 


            {delimiter}Here is the expression information of the expression you received. Pay particular attention to its prefix expressions and infix expressions:
            {dict2block(expression_dictionary)}

            The operator symbol library has a total of ['add', 'mul', 'sub', 'div', 'n4', 'n3', 'n2', 'sqrt', 'sin', 'cos', 'exp', 'log', 'arcsin"], which stand for addition, subtraction, multiplication, division, quadratic, cubic, square, open square, sin, cos, exponential function, logarithmic function, and inverse sine function, respectively.
            
            Your should follow the following thinking process one by one: 
            1. What is the combination method of this symbol? You should give out its detailed operation method. 
            2. How can you hierarchy this expression and break it down into multiple more easily understandable units? Specifically, you can decompose the expression layer by layer from bottom to top, combining some symbols together and treating them as a whole symbol to simplify the expression. 
            Give out your detailed decomposition process. 
            3. Finally, you should summary your reasoning process, and give out a pieces of python dictionary like this(You should output aone piece of combination with 'is_final_expression' is True, and when your combination use other combination as a element in its infix and prefix expression, 
            you should use the overall name of the element combination, if the expression is too simple, you can just give out one piece of combination with 'is_final_expression' is True):

            {delimiter} Combination k(where k represents the number of the combination, starting from 1)
            {dict2block(combination_dictionary)} 
            (...)
            
            {delimiter}Here is Combination 1 case of formula 'exp(-theta**2/2)/sqrt(2*pi)' that you can reference it to output in this format. This is the innermost combination, on top of which more combinations will be put together, and so on, until the final combination is put together to form the combination of the whole formula.
            {dict2block(example_combination)} 
            (...)
            
            Finally, you can output your refined knowledges for each python dictionary you give out.
            Your should follow the following thinking process one by one to give out your refined knowledge(your refined knowledge should reflect these contents): 
            1. What is the combination method?  You should give out its infix expression and explain the operation of it.
            2. what is the physical meaning of the combination? Please note that there maybe free const symbols in the symbol dictionary you received, please check carefully and give a reasonable expaination to them.
            3. What is the physical meaning of this combination? 
            4. How can this physical meaning be further described more concisely?
            
            {delimiter}Here is a example of knowledge, you must keep its output format and give your final output with a python dict. 
            
            {delimiter} Knowledge k(where k represents the number of the knowledge, starting from 1, the number of the knowledge should be the same as the number of the combination you give out)
            {dict2block(knowledge_dictionary)} 
            (...)
            
            {delimiter}Here is Knowledge 1 case of formula 'exp(-theta**2/2)/sqrt(2*pi)' that you can reference it to output in this format. It corresponds to Combination 1 above and is the knowledge abstracted from it.
            {dict2block(example_knowledge)} 
            (...)

            You can stop reasoning once you know how to correct these errors. Your final output should follow the thinking step above, with format "Final reflection: <reflection...>".
            """
        )
        reflect_message = reflect_message.replace("            ", "")
        message = [
            SystemMessage(content=self.system_message),
            AIMessage(content=reflect_message),
        ]
        return message