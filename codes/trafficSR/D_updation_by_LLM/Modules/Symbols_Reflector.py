from langchain.schema import AIMessage, SystemMessage
import textwrap
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_utils import *

# from codes.trafficSR.D_updation_by_LLM.Modules.Knowledge_Pool import Knowledge

delimiter = "####"
example_knowledge = {
    "source": "LLM",
    "content": "LLM agent suggests that the difference between the following distance $s$ and the safe following distance $s_{safe}$ can be used to form a distance influence factor to describe the impact of distance factors on the final following behavior. The calculation of the distance influence factor then uses the following distance $s$ and the safe following distance $s_{safe}$. The calculation method is to first divide the safe following distance from the true following distance and then square it to further expand the influence of the symbol to obtain the combined symbol $(s_{safe}/s)^2$."
}


class Symbols_Reflector:
    def __init__(self):
        self.initialize_system_message_for_symbol_reflector()

    def initialize_system_message_for_symbol_reflector(self):
        self.system_message = textwrap.dedent(
            f'''
            You are a reflection assistant who is responsible for reflect the symbol dictionary output from another reinforce learning agent.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat until you decide to output your final answer>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message = self.system_message.replace("            ", "")

    def initialize_system_message_for_expression_reflector(self):
        self.system_message = textwrap.dedent(
            f'''
            You are a reflection assistant who is responsible for reflect the expression output from another reinforce learning agent.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat until you decide to output your final answer>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message = self.system_message.replace("            ", "")

    def reflect_symbol(self, symbol_dictionary: dict, symbol_library: dict):
        symbol_used_list = list(set(symbol_dictionary['prefix_expression']))
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
            You should give a reflection on the symbol dictionary you received, then finally output a refined knowledge. The symbol you received can fit the car following model well when combined with other symbols. 
            You should explain why it can be well used to combine the car following model, based on the physical meaning of the basic symbol used in this symbol and the combination structure of this symbol.
            {delimiter}Here is the symbol dictionary you received:
            {dict2block(symbol_dictionary)}

            {delimiter}Here is the symbol information of symbols in this symbol dictionary:
            {dict2block(symbol_used_dict)}

            Your should follow the following thinking process one by one: 
            1. What is the combination method of this symbol?  You should give out its infix expression and explain the operation of it.
            2. what is the physical meaning of the symbol used to combine this new symbol? Please note that there maybe free const symbols in the symbol dictionary you received, please check carefully and give a reasonable expaination to them.
            3. What is the physical meaning of this new symbol in this combination method? 
            4. How can this physical meaning be further described as more concise and in line with the car-following scenario?
            
            Then output your final output with a piece of refined knowledge, the name of the given symbol dictionary {symbol_dictionary['name']} shouldn't appear in your output, cause your output should be universal knowledge which can instruct to combine new symbols.
            {delimiter}Here is a example of knowledge, you must keep its output format and give your final output with a python dict.
            {dict2block(example_knowledge)} 

            You can stop reasoning once you know how to correct these errors. Your final output should follow the thinking step above, with format "Final reflection: <reflection...>".
            """
        )
        reflect_message = reflect_message.replace("            ", "")
        message = [
            SystemMessage(content=self.system_message),
            AIMessage(content=reflect_message),
        ]
        return message
