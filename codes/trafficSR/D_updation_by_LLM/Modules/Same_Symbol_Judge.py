import textwrap
from langchain.schema import HumanMessage, SystemMessage
from codes.trafficSR.D_updation_by_LLM.Modules.defaults import *
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_utils import *

delimiter = "####"
true = True
false = False


class Same_Symbol_Judge():
    def __init__(self, llm, save_path):
        self.pattern = r'\#(.*?)\#'
        self.llm = llm
        self.save_path = save_path
        self.initialize_system_message()
        self.initialize_format_message()

    def add_prompt(self, prompt, filename):
        '''
        save str prompt to a file
        '''
        if type(prompt) != str:
            prompt = f"## message type: {type(prompt)}\n" + prompt.content
        else:
            prompt = f"## message type: HumanMessage\n" + prompt
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(prompt + '\n')

    def initialize_system_message(self):
        self.system_message = textwrap.dedent(
            f'''
            You are a judgment assistant who is responsible for checking if two infix expressions represent the same meaning.
            Your response should use the following format:
            <reasoning>
            <reasoning>
            <repeat until you decide to output your final answer>
            Response to user:
            {delimiter} <Your final output>
            Make sure to include {delimiter} to seperate every step.
            '''
        )
        self.system_message = self.system_message.replace("            ", "")

    def extract_infix_expression_of_target(self, AI_answer): # 从##之间提取内容
        return extract_text_between_dollars(self.pattern, AI_answer)

    def generate_info_message(self, symbol_library):
        self.info_message = textwrap.dedent(
            f"""
            The original symbol library that you can refer to is:
            {symbol_library}
            Please judge if the two infix expressions below represent the same meaning (that is to say, they are actually the same symbol):
            """
        )
        self.info_message = self.info_message.replace("            ", "")

    def generate_infix_expression_message(self, fewshot_result, new_symbol_dictionary, env):
        # 利用大模型来进行推理
        self.infix_expression_of_target = self.extract_infix_expression_of_target(fewshot_result['AIMessage'])[0]
        new_symbol_program = env.programs.set_program(new_symbol_dictionary["prefix_expression"])
        self.infix_expression_of_new_symbol = new_symbol_program.get_infix_sympy(do_simplify=True)
        self.infix_expression_message = textwrap.dedent(
            f"""
            1. The infix expression of the symbol that the example wants to combine is:
            {self.infix_expression_of_target}
            2. The infix expression of a newly combined symbol is:
            {self.infix_expression_of_new_symbol}
            """
        )
        self.infix_expression_message = self.infix_expression_message.replace("            ", "")

    def initialize_format_message(self):
        self.format_message = textwrap.dedent(
            f"""
            Please analyze and compare these two infix expressions based on the symbol library. 
            Finally, provide a judgment on whether these two expressions are equivalent.
            The judgement should be output with the format like this:
            {dict2block(DEFAULT_SAME_MEANING_EXPRESSION_JUDGE)}
            Key "equivalent_expressions" should provide a True or False answer based on the equivalence of two expressions.
            Key "reason" should include the reason for your judgment.
            You can stop reasoning once you know how to make judgment.
            Your final output should only contain a python dictionary of the result of judgement.
            """
        )
        self.format_message = self.format_message.replace("            ", "")

    def generate_prompt(self, symbol_library, fewshot_result, new_symbol_dictionary, env):
        self.generate_info_message(symbol_library)
        self.generate_infix_expression_message(fewshot_result, new_symbol_dictionary, env) # 主要判断两个中缀表达式是否相等
        message = [
            SystemMessage(content=self.system_message),
        ]
        human_message = self.info_message
        human_message += self.infix_expression_message
        human_message += self.format_message
        message.append(HumanMessage(content=human_message))
        return message

    def decode_response(self, response):
        final_output = response.split(delimiter)[-1]
        response_dictionary_str = extract_python_code_blocks(final_output)
        if isinstance(response_dictionary_str, ValueError):
            return None, "There should be at least one python code block in this str.", None

        response_dictionary = []
        eval(f"response_dictionary.append({response_dictionary_str})")
        equivalent_dictionary = response_dictionary[0]

        error_list = []
        type_dictionary = {
            "equivalent_expressions": bool,
            "reason": str,
        }
        # check key and type
        for key in type_dictionary.keys():
            if key not in equivalent_dictionary.keys():
                error_list.append(f"Key {key} is missing in your output dictionary.")
            else:
                if type(equivalent_dictionary[key]) != type_dictionary[key]:
                    error_list.append(f"Type of {key} is not correct, it should be {type_dictionary[key]}.")

        error_message = ""
        if len(error_list) == 0:
            return equivalent_dictionary["equivalent_expressions"], error_message, None
        else:
            check_message = textwrap.dedent(
                f"""
                The output dictionary of equivalence judgement is:
                {equivalent_dictionary}
                After checking, here are the errors I found in the output dictionary:
                """
            )
            for error in error_list:
                error_message += error
            check_message += error_message
            check_message += textwrap.dedent(
                f"""
                You should correct the errors and output a new dictionary of judgement on whether these two expressions are equivalent.
                The new symbol dictionary should be output with the format like this:
                {dict2block(DEFAULT_SAME_MEANING_EXPRESSION_JUDGE)}
                You can stop reasoning once you know how to correct these errors.
                Your final output should only contain a python dictionary of the result of judgement.
                """
            )
            check_message = check_message.replace("                ", "")
            check_messages = [
                SystemMessage(content=self.system_message),
                HumanMessage(content=check_message),
            ]
            return equivalent_dictionary["equivalent_expressions"], error_message, check_messages

    def judge_equivalent_expressions(self, symbol_library, fewshot_result, new_symbol_dictionary, env, max_try_num=2):
        equivalent_message = self.generate_prompt(symbol_library, fewshot_result, new_symbol_dictionary, env)
        for m in equivalent_message:
            self.add_prompt(m, self.save_path)
        equivalent_response = ""
        for chunk in self.llm.stream(equivalent_message):
            equivalent_response += chunk.content
            print(chunk.content, end="", flush=True)
        self.add_prompt(equivalent_response, self.save_path)

        tried_num = 0
        while tried_num < max_try_num:
            bool_equivalent_expressions, error_message, check_messages = self.decode_response(equivalent_response)
            print(f"################# The {tried_num}-th trial of combination equivalent_judge #################")
            if bool_equivalent_expressions == True:
                print("The prefix expression is equivalent to the given example prefix.")
                return True, self.infix_expression_of_target, self.infix_expression_of_new_symbol
            elif bool_equivalent_expressions == False:  # regenerate symbol
                print("The prefix expression is not equivalent to the given example prefix.")
                return False, self.infix_expression_of_target, self.infix_expression_of_new_symbol
            elif error_message == "There should be at least one python code block in this str.":  # directlt regenerate judgement
                equivalent_response = ""
                for chunk in self.llm.stream(equivalent_message):
                    equivalent_response += chunk.content
                    print(chunk.content, end="", flush=True)
                self.add_prompt(equivalent_response, self.save_path)
            else:  # error response for equivalent answer, need to correct
                equivalent_response = ""
                for chunk in self.llm.stream(check_messages):
                    equivalent_response += chunk.content
                    print(chunk.content, end="", flush=True)
                self.add_prompt(equivalent_response, self.save_path)
            tried_num += 1

        if tried_num >= max_try_num:  # avoid error loop
            print(f"################# Combination comparison tried too many times, failed #################")
            return False, self.infix_expression_of_target, self.infix_expression_of_new_symbol
