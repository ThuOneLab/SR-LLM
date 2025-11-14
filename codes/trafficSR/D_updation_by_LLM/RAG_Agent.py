from codes.trafficSR.D_updation_by_LLM.Agent import Agent
from codes.trafficSR.D_updation_by_LLM.Modules.Directly_Builder import Directly_Builder
from codes.trafficSR.D_updation_by_LLM.Modules.Expressions_Reflector import Expressions_Reflector,Expressions_Reflector_Random
from codes.trafficSR.D_updation_by_LLM.Modules.Knowledge_Pool import Knowledge_Pool, Knowledge
from codes.trafficSR.D_updation_by_LLM.Modules.Model_Combiner import Model_Combiner
from codes.trafficSR.D_updation_by_LLM.Modules.Prompt_Generator import Prompt_Generator,Prompt_Generator_Feyn,Prompt_Generator_Random
from codes.trafficSR.D_updation_by_LLM.Modules.Same_Symbol_Judge import Same_Symbol_Judge
from codes.trafficSR.D_updation_by_LLM.Modules.Symbols_Decoder import Symbols_Decoder
from codes.trafficSR.D_updation_by_LLM.Modules.Symbols_Reflector import Symbols_Reflector
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_utils import *
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_default_setting import *

class RAG_AGENT(Agent):
    def __init__(self, trainResultFolder=None, agent_config=DEFAULT_AGENT_CONFIG, port="15732", address="172.22.0.1", fewshot_num=0, reflection_num=0, extend_num=0, time=None, memory_path="memory_idm", only_directly_built=True,bool_is_feynman=False,bool_is_random=False):
        super(RAG_AGENT, self).__init__(trainResultFolder=trainResultFolder, agent_config=agent_config, port=port, address=address)
        self.knowledge_pool = Knowledge_Pool(memory_path) # 用于存储知识,已解决
        self.directly_builder = Directly_Builder()
        self.prompt_generator = Prompt_Generator_Feyn() if bool_is_feynman else Prompt_Generator()
        if bool_is_random:
            self.prompt_generator = Prompt_Generator_Random()
        self.symbols_decoder = Symbols_Decoder(bool_is_feynman=bool_is_feynman)
        self.symbol_reflector = Symbols_Reflector()
        self.expression_reflector = Expressions_Reflector()
        if bool_is_random:
            self.expression_reflector = Expressions_Reflector_Random()
        self.model_combiner = Model_Combiner()
        self.bool_is_feynman = bool_is_feynman
        
        time_path=os.path.join(self.llm_log_folder, f"{time}/")
        if not os.path.exists(time_path):
            os.makedirs(time_path)
        self.save_path = os.path.join(time_path,f'fewshotnum{fewshot_num}_reflectionnum{reflection_num}_extendnum{extend_num}.md')
        if not os.path.isfile(self.save_path):
            with open(self.save_path, 'w') as f:
                print(f"Create a new history_fewshot file: {self.save_path}")
                
        self.reflection_path = os.path.join(time_path,f'reflectionnum{reflection_num}.txt')
        if not os.path.isfile(self.reflection_path):
            with open(self.reflection_path, 'w') as f:
                print(f"Create a new history_reflection file: {self.reflection_path}")
                
        self.same_symbol_judge = Same_Symbol_Judge(self.llm, self.save_path)
        self.only_directly_built = only_directly_built

    # 核心函数
    def generate_new_symbol(self, symbol_info, library_info, env, fewshot_num=0, expression_info=None, max_try_num=3):
        query_symbol_description = "$" + symbol_info['name'] + "$: " + symbol_info['description']
        # retrieve knowledge from the knowledge pool
        fewshot_results = self.knowledge_pool.retrieve_knowledge(query=query_symbol_description, fewshot_num=fewshot_num)
        count = 0
        '''检查1：需要找到足够的例子'''
        # TODO:可能可以进一步改进，如果相似度过低，就不使用那么多
        # while len(fewshot_results) < fewshot_num and count < max_try_num:
        #     # print("Not shot examples found, find examples for the new symbol again. Because knowledge changed!!")
        #     fewshot_results = self.knowledge_pool.retrieve_knowledge(query=query_symbol_description, fewshot_num=fewshot_num)
        #     count += 1

        '''检查2：检查是否可以直接构建'''
        # check if the symbol can be built directly and not been achieved
        not_achieved = self.directly_builder.not_achieved(fewshot_results, library_info)  # have not been achieved
        can_be_built_directly = self.directly_builder.can_be_built_directly(fewshot_results, symbol_info, library_info, ) # can be built directly
        build_directly = [not_achieved[i] and can_be_built_directly[i] for i in range(len(not_achieved))] # can be built directly and have not been achieved
        
        '''能直接构建的有哪些knowledge'''
        directly_targetSymbols, directly_built_knowledge_ids = self.directly_builder.can_be_build_directly_examples(fewshot_results, build_directly)
        directly_built_combinations = []
        if len(directly_targetSymbols) == 0:
            print("No directly built target symbols.")

        '''generate message used for constructing new symbols'''
        example_messages, expression_md, message = self.prompt_generator.generate_prompt(symbol_info, library_info, fewshot_results=fewshot_results, not_achieved=not_achieved, build_directly=build_directly, expression_info=expression_info, )
        for m in message:
            self.add_prompt(m, self.save_path)
        new_symbol_dictionaries = []

        '''生成新符号'''
        # self.only_directly_built and 
        # if (len(directly_targetSymbols) > 0) or not self.only_directly_built:  # for find idm, only need to generate useful one(only directly built).
        if True:  # for random, need to generate all
            for i in range(len(message) - 1): #遍历所有的例子
                # generate combination answer
                response = ""
                print(f"RAG Example:\n", example_messages[i])
                if expression_md != "":
                    print(f"Expression Example:\n", expression_md)
                # httpx.RemoteProtocolError: peer closed connection without sending complete message body (incomplete chunked read)’
                # 生成回答
                for chunk in self.llm.stream([message[0], message[i + 1]]):
                    response += chunk.content
                    print(chunk.content, end="", flush=True)
                self.add_prompt(response, self.save_path)

                # decode the response
                tried_num = 0
                new_symbol_dictionary = []
                while tried_num < max_try_num:
                    print(f"################# The {tried_num}-th trial of combination generation #################")
                    '''检查3：检查生成的符号是否合法'''
                    # decode the response
                    new_symbol_dictionary, error_message, \
                        check_messages = self.symbols_decoder.decode_symbol(response,
                                                                            library_info,
                                                                            env,
                                                                            tolerate_existing_symbol=False,
                                                                            directly_targetSymbol=
                                                                            directly_targetSymbols[
                                                                                i] if len(
                                                                                directly_targetSymbols) > 0 else None,
                                                                            example_message=example_messages[i] if len(
                                                                                directly_targetSymbols) > 0 else None,
                                                                            knowledge_target_names=
                                                                            self.knowledge_pool.knowledge_target_names if len(
                                                                                directly_targetSymbols) == 0 else None, # None for directly built, 0 for fuzzy built(不能在fuzzy中构建出所需构建的知识)
                                                                            )

                    if check_messages is None:
                        print("\nOutput dictionary is semantically legal.")
                        break
                    #检查5：字典内容不合法，重新生成
                    elif check_messages == "There should be at least one python code block in this str.":
                        print(check_messages + " So regenerate combination.\n")
                        response = ""
                        for chunk in self.llm.stream([message[0], message[i + 1]]):
                            response += chunk.content
                            print(chunk.content, end="", flush=True)
                        self.add_prompt(response, self.save_path)
                    # 
                    else:
                        # print(check_messages[-1].content) #too long
                        print(error_message)
                        self.add_prompt(check_messages[-1].content, self.save_path)
                        response = ""
                        for chunk in self.llm.stream(check_messages):
                            response += chunk.content
                            print(chunk.content, end="", flush=True)
                        self.add_prompt(response, self.save_path)
                    tried_num += 1
                if tried_num < max_try_num:  # avoid error loop
                    new_symbol_dictionaries.append(new_symbol_dictionary)
                    if len(directly_targetSymbols) > 0:
                        directly_built_combinations.append(directly_targetSymbols[i])
                else:
                    print(f"################# Combination generation tried too many times, failed #################")

        return new_symbol_dictionaries, directly_built_combinations

    def reflect_new_symbol(self, elite_symbol_dictionary: dict, symbol_library: dict, human_comment: str = None):
        # if elite_symbol_dictionary['performance'] != "Good":
        #     raise ValueError("Only good symbol can be reflected.")
        # if human_comment is None:
        #     human_comment = "Good symbol with excellent fitting performance."
        ai_reflection = "It's a good symbol, so I should learn from it to create new symbols next time."
        message = self.symbol_reflector.reflect_symbol(elite_symbol_dictionary, symbol_library)
        for m in message:
            self.add_prompt(m, self.save_path)
        response = ""
        for chunk in self.llm.stream(message):
            response += chunk.content
            print(chunk.content, end="", flush=True)
        self.add_prompt(response, self.save_path)

        # knowledge = Knowledge(
        #     source="DRL",
        #     key=elite_symbol_dictionary['key'],
        #     content=refined_content,
        #     comment=human_comment,
        #     reflection=ai_reflection,
        # )
        return

    def reflect_new_expression(self, expression_dictionary: dict, symbol_library: dict, human_comment: str = None):
        # if elite_symbol_dictionary['performance'] != "Good":
        #     raise ValueError("Only good symbol can be reflected.")
        if human_comment is None:
            human_comment = "Good symbol with excellent fitting performance."
        ai_reflection = "It's a good symbol, so I should learn from it to create new symbols next time."
        message = self.expression_reflector.reflect_expression(expression_dictionary, symbol_library)
        for m in message:
            self.add_prompt(m, self.save_path)
        response = ""
        for chunk in self.llm.stream(message):
            response += chunk.content
            print(chunk.content, end="", flush=True)
        self.add_prompt(response, self.save_path)

        combinations = []
        knowledges = []

        str_blocks = response.split(delimiter)

        for block in str_blocks:
            if block.startswith(" Combination ") and not block.startswith(" Combination k"):
                eval(f"combinations.append({extract_python_code_blocks(block)})")
            if block.startswith(" Knowledge ") and not block.startswith(" Knowledge k"):
                eval(f"knowledges.append({extract_python_code_blocks(block)})")
                knowledges[-1]['comment'] = human_comment
                knowledges[-1]['reflection'] = ai_reflection

        return combinations, knowledges

        # knowledge = Knowledge(
        #     source="DRL",
        #     key=elite_symbol_dictionary['key'],
        #     content=refined_content,
        #     comment=human_comment,
        #     reflection=ai_reflection,
        # )

    def choose_symbol(self, symbol_library: dict, env):
        # choose a symbol from the symbol library
        message = self.model_combiner.generate_prompt_to_choose_symbol(symbol_library)
        for m in message:
            self.add_prompt(m, self.save_path)

        response = ""
        for chunk in self.llm.stream(message):
            response += chunk.content
            print(chunk.content, end="", flush=True)
        self.add_prompt(response, self.save_path)

        # decode the response
        while True:
            new_symbol_dictionary, check_messages = self.symbols_decoder.decode_symbol(response, symbol_library, env,
                                                                                       tolerate_existing_symbol=True)
            if check_messages is None:
                print("\nOutput dictionary is semantically legal.")
                break
            else:
                print(check_messages[-1].content)
                self.add_prompt(check_messages[-1].content, self.save_path)

                response = ""
                for chunk in self.llm.stream(check_messages):
                    response += chunk.content
                    print(chunk.content, end="", flush=True)
                self.add_prompt(response, self.save_path)

        return new_symbol_dictionary

    def combine_model(self, library_info):
        message = self.model_combiner.generate_prompt_to_combine_model(library_info)
        for m in message:
            self.add_prompt(m, self.save_path)

        response = ""
        for chunk in self.llm.stream(message):
            response += chunk.content
            print(chunk.content, end="", flush=True)
        self.add_prompt(response, self.save_path)

        # decode the response
        # while True:
        #     new_symbol_dictionary, check_messages = self.symbols_decoder.decode_symbol(response, library_info, env)
        #     if check_messages is None:
        #         print("\nOutput dictionary is semantically legal.")
        #         break
        #     else:
        #         print(check_messages[-1].content)
        #         response = ""
        #         for chunk in self.llm.stream(check_messages):
        #             response += chunk.content
        #             print(chunk.content, end="", flush=True)

        # return new_symbol_dictionary

    def update_knowledge_pool(self, combinations, knowledges, library_info):
        for c, k in zip(combinations, knowledges):
            keys = list(set(c["prefix_expression"]))
            final_key = []
            for i, key in enumerate(keys):
                if key in library_info.keys():
                    if library_info[key]["type"] == "combination" or library_info[key]["type"] == "variable":
                        final_key.append("$" + key + "$: " + library_info[key]["description"])
                else:
                    final_key.append("$" + key + "$: ")
                if i < len(keys) - 1:
                    final_key.append(", ")
            keys_str = ""
            for key in final_key:
                keys_str += key

            target = c['name']
            target_str = ""
            if target in library_info.keys():
                if library_info[target]["type"] == "combination" or library_info[target]["type"] == "variable":
                    target_str = "$" + target + "$: " + library_info[target]["description"]
            else:
                target_str = "$" + target + "$: "
            
            # 存储到文件，便于查看
            dict_item = {"source": k["source"], "target": target_str, "key": keys_str, "content": k['content'],
                         "comment": k['comment'], "reflection": k['reflection']}
            save_dict_to_text_file(dict_item, self.reflection_path)
            
            # 加入到knowledge_pool
            knowledge = Knowledge(source=k["source"], target=target_str, key=keys_str, content=k['content'], comment=k['comment'], reflection=k['reflection'], )
            trial_num = 0
            while True:
                try:
                    self.knowledge_pool.add_knowledge(knowledge)
                    break
                except:
                    print("Failed to add knowledge, try again.")
                    trial_num += 1
                    if trial_num >= 3:
                        print("Failed to add knowledge for many times, give up.")
                    continue

            self.knowledge_pool.save_target_names()
            