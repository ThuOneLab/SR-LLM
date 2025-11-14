import textwrap
from langchain.schema import HumanMessage, SystemMessage
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken_combinations import Token_combination
from codes.trafficSR.D_updation_by_LLM.Modules.defaults import *
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_utils import *

# Units_Allowed = [[0, 0], [1, 0], [0, 1], [1, -2], (2, -4):"jerk"]
Units_Allowed = {(0, 0): ("dimensionless", "factor"), (1, 0): ("distance", "s"), (0, 1): ("time", "t"),
                 (1, -1): ("speed", "v"), (1, -2): ("acceleration", "a"),
                 (2, -2): ("productSpeed", "v2"), (2, -4): ("productAcceleration", "a2")}


class Symbols_Decoder():
    def __init__(self,bool_is_feynman=True):
        self.initialize_system_message()
        self.bool_is_feynman = bool_is_feynman
        self.new_symbol_need_in_units_allowed = False if self.bool_is_feynman else True #对于feynman，不需要在allowed units里面
    
    def initialize_system_message(self):
        self.system_message = textwrap.dedent(
            f'''
            You are a output checking assistant who is responsible for checking and correcting the output symbol dictionary of another agent.
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

    def decode_symbol(self, response, symbol_library, env, tolerate_existing_symbol=False, directly_targetSymbol=None,
                      example_message=None,
                      knowledge_target_names=None):
        self.env = env
        final_output = response.split(delimiter)[-1]
        new_symbol_dictionary, error_message, check_messages = self.check_new_symbol_errors(final_output, symbol_library,
                                                                                        tolerate_existing_symbol,
                                                                                        directly_targetSymbol,
                                                                                        example_message,
                                                                                        knowledge_target_names)

        return new_symbol_dictionary, error_message, check_messages
    
    def check_key_valid_in_symbol_library(self, symbol_dictionary, error_list):
        type_dictionary = {
            "name": str,
            "description": str,
            "units": list,
            "prefix_expression": list,
        }
        all_keys_exist = True # whether all keys exist in the output dictionary
        for key in type_dictionary.keys():
            if key not in symbol_dictionary.keys():
                error_list.append(f"Key {key} is missing in your output dictionary.")
                all_keys_exist = False
            else:
                if symbol_dictionary[key] is None or len(symbol_dictionary[key]) == 0:  # have no tokens in combination
                    error_list.append(f"Key {key} should not be empty.")
                    all_keys_exist = False
                if type(symbol_dictionary[key]) != type_dictionary[key]:
                    error_list.append(f"Type of {key} is not correct, it should be {type_dictionary[key]}.")
                    all_keys_exist = False
        return all_keys_exist, error_list
    
    def check_symbol_in_library(self, symbol_dictionary, symbol_library, error_list):
        all_symbol_available = True
        if "prefix_expression" in symbol_dictionary.keys():
            prefix_expression_with_basic_symbols = []
            symbol_dictionary['prefix_expression'] = flatten_list(symbol_dictionary['prefix_expression'])
            '''检查回答中符号都在库中'''
            for symbol in symbol_dictionary['prefix_expression']:
                if symbol not in symbol_library.keys():
                    error_list.append(
                        f"Symbol {symbol} is not in the variable list of the library, please check if you can use alternative symbols to represent {symbol}. Your corrected symbol should be as consistent as possible with the symbol before correction, and make sure the new symbol is not in the symbol library either.")
                    all_symbol_available = False
            '''检查回答中所有符号都在库中——这里把组合的前缀表达式展平成list，再进行检查'''
            if all_symbol_available:
                for symbol in symbol_dictionary['prefix_expression']:
                    prefix_expression_with_basic_symbols.append(symbol_library[symbol]['prefix_expression'])
                symbol_dictionary["prefix_expression"] = flatten_list(prefix_expression_with_basic_symbols)
                # symbol_dictionary["prefix_expression"] = symbol2prefixlist(symbol_dictionary) # same function, but slower
                for symbol in symbol_dictionary['prefix_expression']:
                    if symbol not in symbol_library.keys():
                        error_list.append(
                            f"Symbol {symbol} is not in the variable list of the library, please check if you can use alternative symbols to represent {symbol}. Your corrected symbol should be as consistent as possible with the symbol before correction, and make sure the new symbol is not in the symbol library either.")
                        all_symbol_available = False
        return all_symbol_available, error_list
    
    def check_new_symbol_too_long(self, all_symbol_available,symbol_dictionary, error_list):
        max_time_step_ratio = 0.7
        if len(symbol_dictionary["prefix_expression"]) > int(self.env.max_time_step * max_time_step_ratio):
            error_list.append(
                f"The total number of characters in the prefix expression for the new symbol cannot exceed {int(self.env.max_time_step * max_time_step_ratio)}. " + \
                f"Currently, the number of characters in the new symbol is {len(symbol_dictionary['prefix_expression'])}, and you need to adjust the combination to regenerate the symbol.")
            all_symbol_available = False
        return all_symbol_available, error_list
    
    def check_new_symbol_already_in_library(self, symbol_dictionary, symbol_library, error_list, env, all_symbol_available):
        def library2prefixlist(library):
            prefix_list = []
            for key in library.keys():
                prefix_list.append(library[key]['prefix_expression'])
            return prefix_list
        
        # new symbol名称已经在库中
        if symbol_dictionary['name'] in self.env.library.all_tokens_id_dict.keys():
            error_list.append(
                f"The new symbol name {symbol_dictionary['name']} is already in the symbol library, please generate a new name.")
        # 或者前缀表达式已经在库中（可能不是这个同样名称）
        elif all_symbol_available and symbol_dictionary["prefix_expression"] in library2prefixlist(symbol_library):
            error_list.append(
                f"The new symbol's prefix expression {symbol_dictionary['prefix_expression']} is already in the symbol library, please generate a symbol with different prefix expression.")
        return error_list
   
    def check_arity_equal_to_zero(self, symbol_dictionary, error_list):
        sum_arity = 1
        for symbol in symbol_dictionary['prefix_expression']:
            sum_arity += self.env.library.all_token_info_table.arity_table[self.env.library.all_tokens_id_dict[symbol]] - 1
        if sum_arity != 0:
            error_list.append(
                f"The combination method of the new symbol is illegal, please check the arity of the prefix expression, make sure it represents a legal mathematical expression. ")
        return error_list
    
    def units_must_in_allowed(self, symbol_dictionary, error_list,real_units):
        symbol_dictionary['units'] = real_units
        error_list.append(
            f"Your new symbol's physical unit must only among this: " + \
            f"[1, 0](m, represent the distance item), [0, 1] (s, represent the time item), [1, -1] (m/s, represents the speed item), [1, -2] (m/s^2, represent the acceleration item), [0, 0] (dimensionless, represents the factor influence the car-following behavior), " + \
            f"[2, -2] (m^2/s^2, represents the relative change in speed between the ego vehicle and the preceding vehicle), [2, -4] (m^2/s^4, represents the overall dynamic response capability of the vehicle).  " + \
            f"Your new symbol's actual physical unit is {real_units}, not in the allowed unit list. " + \
            f"Please change the combination method and regenerate new symbol with unit within the allowed unit list.")

    def check_normalized_name(self, symbol_dictionary, error_list, bool_need_example_message, directly_targetSymbol=None, knowledge_target_names=None):
        if directly_targetSymbol is not None:  # directly_built
            if symbol_dictionary['name'] != directly_targetSymbol:
                bool_need_example_message = True
                error_list.append(
                    f"Based on examples, the name of the new symbol should be {directly_targetSymbol}, " + \
                    f"not current new symbol name {symbol_dictionary['name']}. " + \
                    f"This means that you may not have followed the combination method of examples. " + \
                    f"Please check: If the combination method of the example is implemented, only a name change is needed; " + \
                    f"If it is not combined according to the example, please use the combination method of the example again.")
            # else:  # TODO:need to avoid same name but different prefix_expression——essentially infix_expression
        
        '''avoid the Fuzzy Generation name conflict with directly built target symbols'''
        if knowledge_target_names is not None:  
            if symbol_dictionary['name'] in knowledge_target_names:
                error_list.append(
                    f"Since we cannot directly combine examples, new symbol cannot occupy the target symbol names of examples in the knowledge base. " + \
                    f"The target names include {knowledge_target_names}. "
                    f"{symbol_dictionary['name']} is in the required names for examples in the knowledge base, so you need to adjust the combination method and change the name."
                )
        return error_list, bool_need_example_message
    
    def check_and_normalize_for_car_following(self, symbol_dictionary, error_list, bool_need_example_message, real_units, directly_targetSymbol=None, knowledge_target_names=None):
        if tuple(real_units) not in Units_Allowed:
            self.units_must_in_allowed(symbol_dictionary, error_list,real_units)
        else:  # units are legal and belong to the Units allowed, then human normalize
            '''normalize the unit'''
            symbol_dictionary['units'] = real_units
            
            '''normalize the name factor_v_ratio: need real_units'''
            name_meaning = extract_text_between_dollars(pattern = r'_(.*)', text=symbol_dictionary['name']) # use the original unit to extract the name, mainly for _y
            if len(name_meaning) == 0:
                symbol_dictionary['name'] = f"{Units_Allowed[tuple(real_units)][1]}"
                error_list.append(
                    f"Name requires the naming of symbols as 'x_y': " + \
                    f"x reflects the unit of symbol('s_y' represnts [1, 0], 't_y' represents [0, 1], 'v_y' represents [1, -1],'a_y' represents [1, -2], 'factor_y' represents [0, 0], 'v2_y' representes [2, -2], 'a2_y' represents [2, -4]), and y represents the condense meaning of this symbol in the driving scenario(but not calculation method like x_v_div_t)." + \
                    f"Please pay attention to the name format in the subsequent generation.")
            else:
                symbol_dictionary['name'] = f"{Units_Allowed[tuple(real_units)][1]}_{name_meaning[0]}"
                error_list, bool_need_example_message = self.check_normalized_name(symbol_dictionary, error_list, bool_need_example_message, directly_targetSymbol, knowledge_target_names)
                        
            '''normalize the description'''
            represent_meaning = extract_text_between_dollars(pattern = r'unit which represents\s(.*)', text=symbol_dictionary['description'])
            if len(represent_meaning) == 0:
                error_list.append(
                    f"The description should follow the format like this: 'The symbol in ...(distance/time/speed/acceleration/dimensionless/productSpeed/productAcceleration) unit which represents ...'. " + \
                    f"Please pay attention to the description format in the subsequent generation.")
            else:
                symbol_dictionary[
                    'description'] = f"The symbol in {Units_Allowed[tuple(real_units)][0]} unit which represents {represent_meaning[0]}"
        return error_list, bool_need_example_message
    
    def check_and_normalize_for_feynman(self, symbol_dictionary, error_list, bool_need_example_message, real_units, directly_targetSymbol=None, knowledge_target_names=None):
        '''normalize the unit'''
        symbol_dictionary['units'] = real_units
                        
        '''normalize the description'''
        represent_meaning = extract_text_between_dollars(pattern = r'which represents\s(.*)', text=symbol_dictionary['description'])
        if len(represent_meaning) == 0:
            error_list.append(
                f"The description should follow the format like this: 'The symbol which represents ...'. " + \
                f"Please pay attention to the description format in the subsequent generation.")
        return error_list, bool_need_example_message
    
    def check_new_symbol_errors(self, final_output, symbol_library, tolerate_existing_symbol=False, directly_targetSymbol=None, example_message=None, knowledge_target_names=None):
        '''提取python代码块'''
        new_symbol_dictionary_str = extract_python_code_blocks(final_output)
        if isinstance(new_symbol_dictionary_str, ValueError):
            return None, "", "There should be at least one python code block in this str."
        new_symbol_dictionary = []
        eval(f"new_symbol_dictionary.append({new_symbol_dictionary_str})")
        symbol_dictionary = new_symbol_dictionary[0]

        error_list = []

        def symbol2prefixlist(symbol_dictionary):
            prefix_list = []
            for key in symbol_dictionary["prefix_expression"]:
                token_idx = self.env.library.all_tokens_id_dict[key]
                prefix_list += self.env.all_tokens[token_idx].prefix_expression
            return prefix_list

        '''检查是否在回答中，所有key都存在，类型都正确'''
        all_keys_exist, error_list = self.check_key_valid_in_symbol_library(symbol_dictionary, error_list)

        '''检查回答中符号都在库中'''
        all_symbol_available, error_list = self.check_symbol_in_library(symbol_dictionary, symbol_library, error_list)
        
        '''检查回答的组合是否过长，超出最大长度的0.7倍'''
        all_symbol_available, error_list = self.check_new_symbol_too_long(all_symbol_available,symbol_dictionary, error_list)

        '''check if the new symbol is already in the symbol library'''
        if all_keys_exist and not tolerate_existing_symbol:
            error_list = self.check_new_symbol_already_in_library(symbol_dictionary, symbol_library, error_list, self.env, all_symbol_available)
        
        '''check if the new symbol's combination method is legal, name_description_prefix_units'''
        bool_need_example_message = False
        if all_keys_exist and all_symbol_available:
            '''need arity=0'''
            error_list = self.check_arity_equal_to_zero(symbol_dictionary, error_list)

            def if_units_number(unit):
                if not isinstance(unit, list):
                    return False
                for u in unit:
                    if type(u) == str:
                        return False
                return True

            if if_units_number(symbol_dictionary['units']) is False:
                error_list.append(f"The units of the new symbol should be a list of numbers, not {symbol_dictionary['units']}.")
            else:
                '''check if the physical unit is correct'''
                tok = Token_combination(
                    all_tokens=self.env.all_tokens,
                    library=self.env.library,
                    name=symbol_dictionary['name'],
                    type='combination',
                    id=0,
                    prefix_expression=symbol_dictionary['prefix_expression'],
                    phy_units=symbol_dictionary['units'], # TODO:有可能生成错误需要进一步判断：["units_n",  # Placeholder indicating actual unit for n."units_alpha"  # Placeholder indicating actual unit for alpha.]
                    description=symbol_dictionary['description'],
                )
                if not tok.legal:  # units are not legal, have error
                    error_list.append(
                        f"Physical unit error happens in this prefix expression of combination, please check the prefix expression, make sure the required units for operators are correct. ")
                # TODO: 这里的单位检查需要进一步细化，要根据每种知识库可能存在不同的单位要求
                elif tok.phy_units is not None:  # units are legal, need to check it belongs to the Units allowed
                    units_top_num=7 if self.bool_is_feynman else 2
                    real_units = [int(unit) for unit in tok.phy_units.tolist()[:units_top_num]]
                    if self.new_symbol_need_in_units_allowed: #用于跟驰模型
                        error_list, bool_need_example_message = self.check_and_normalize_for_car_following(symbol_dictionary, error_list, bool_need_example_message, real_units, directly_targetSymbol, knowledge_target_names)
                    else: #用于feynman
                        error_list, bool_need_example_message = self.check_and_normalize_for_feynman(symbol_dictionary, error_list, bool_need_example_message, real_units, directly_targetSymbol, knowledge_target_names) #好像名称很难弄成一样

        '''返回结果，并且输出错误信息'''
        error_list = [f"{idx + 1}." + error_list[idx] + "\n" for idx in range(len(error_list))]
        if len(error_list) == 0:
            return symbol_dictionary, "", None
        else:
            # need to have example info
            error_message, check_messages = self.generate_check_message(symbol_dictionary, symbol_library, error_list, example_message if bool_need_example_message else None)
            return None, error_message, check_messages

    # 用error生成check_message
    def generate_check_message(self, symbol_dictionary, symbol_library, error_list, example_message=None):
        error_message = ""
        check_message = textwrap.dedent(
            f"""
            The output dictionary of new symbol you received is:
            {symbol_dictionary}
            The original symbol library is:
            {symbol_library}
            """
        )
        if example_message is not None:  # add example message
            check_message += textwrap.dedent(
                f"""
            The following are some examples of how to construct a new symbol for constructing a car-following model based on the given symbols.
            You need refer to those examples to form new symbols based on the above given symbols.
                """
            )
            check_message += example_message
        check_message += textwrap.dedent(
            f"""
            After checking based on symbol library, units and output format, here are the errors I found in the new symbol's output dictionary:
            """
        )
        for error in error_list:
            error_message += error
        check_message += error_message
        check_message += textwrap.dedent(
            f"""
            You should correct the errors and output a new symbol dictionary.
            The new symbol dictionary should be output with the format like this:
            {dict2block(DEFAULT_SYMBOL_FORMAT)}
            You can stop reasoning once you know how to correct these errors.
            Your final output should only contain a python dictionary of the new symbol.
            """
        )
        check_message = check_message.replace(interval, "")
        check_messages = [
            SystemMessage(content=self.system_message),
            HumanMessage(content=check_message),
        ]
        return error_message, check_messages
