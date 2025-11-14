import copy
import textwrap
import time

# from langchain.callbacks import OpenAICallbackHandler
from langchain.callbacks import OpenAICallbackHandler
# from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from rich import print

from codes.trafficSR.E_train_codes.train_monitor.SRvisualization import simplify_True_length_limit
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_utils import *
from codes.trafficSR.D_updation_by_LLM.Modules.Agent_default_setting import *

class Agent():
    '''
    LLM Agent class for Symbolic Regression
    '''

    def __init__(self, trainResultFolder=None, agent_config=DEFAULT_AGENT_CONFIG, port="15732", address="172.22.0.1") -> None:
        # set the config
        self.config = agent_config
        self.port = port
        self.address = address
        self.api_type = self.config['api_type']
        assert self.api_type in ['openai'], f"api {self.api_type} not supported"
        self.model_type = self.config['model_type']
        assert self.model_type in ['gpt-3.5-turbo-16k-0613', 'gpt-4o-mini',
                                   'gpt-4-1106-preview'], f"model_type {self.model_type} not supported"
        self.api_key = self.config['api_key']
        self.initial_trial = self.config['initial_trial']
        self.sleep_time = self.config['sleep_time']

        # build the path variable
        self.build_path_var()

        # create the LLM
        self.llm = ChatOpenAI(
            temperature=self.config['temperature'],
            callbacks=[
                OpenAICallbackHandler()
            ],
            model_name=self.config['model_type'],
            # max_tokens=2000,
            max_tokens=4096,
            request_timeout=60,
            streaming=True,
            base_url='https://xiaoai.plus/v1'
        )
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + "/"
        # md folder
        self.md_save_folder = repo_root + "codes/trafficSR/D_updation_by_LLM/md_save"
        if not os.path.exists(self.md_save_folder):
            os.makedirs(self.md_save_folder)

        # md files of basic messages of human
        self.basic_folder = repo_root + "codes/trafficSR/D_updation_by_LLM/md_basic"
        if trainResultFolder is not None:
            if not os.path.exists(self.basic_folder):
                os.makedirs(self.basic_folder)

        # llm log folder
        self.llm_log_folder = repo_root + "codes/trafficSR/D_updation_by_LLM/llm_log/"
        if not os.path.exists(self.llm_log_folder):
            os.makedirs(self.llm_log_folder)

        # train result folder
        self.trainResultFolder = trainResultFolder

        # max iteraction nums with the LLM
        self.max_iter = 100

        # init memory
        if trainResultFolder is not None:
            self.init_memory()

    def build_path_var(self):
        '''
        Build the path variable for the API
        '''
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_CHAT_MODEL"] = self.model_type
        # os.environ["http_proxy"] = f"http://{self.address}:" + self.port  # 15732
        # os.environ["https_proxy"] = f"http://{self.address}:" + self.port
        # os.environ["all_proxy"] = f"socks5://{self.address}:" + self.port
        # os.environ["ALL_PROXY"] = f"socks5://{self.address}:"+ self.port

    def describeTokens(self,
                       token_args=DEFAULT_TOKEN_ARGS,
                       dcp_args=DEFAULT_TOKEN_DESCRIPTION,
                       unit_names=['m', 's']):
        '''
        Describe the tokens in human readable format
        '''
        output = {}
        # operator_tokens
        output['operator_tokens'] = []
        for ot in token_args['operator_tokens']:
            output['operator_tokens'].append(
                {
                    'token_name': ot,
                    'token_type': 'operator',
                    'length': 1,
                    'description': dcp_args[ot]
                }
            )
        # input variable tokens
        output['variable_tokens'] = []
        for vt in token_args['variable_tokens']:
            vt_physical_units = {}
            for k in range(len(unit_names)):
                unit_name = unit_names[k]
                vt_physical_units[unit_name] = token_args['variable_units'][vt][k]
            output['variable_tokens'].append(
                {
                    'token_name': vt,
                    'token_type': 'input_variable',
                    'length': 1,
                    'description': dcp_args[vt],
                    'physical_units': vt_physical_units
                }
            )

        # fixed const tokens
        output['fixed_const_tokens'] = []
        for fct in token_args['fixed_const_tokens']:
            fct_physical_units = {}
            for k in range(len(unit_names)):
                unit_name = unit_names[k]
                fct_physical_units[unit_name] = token_args['fixed_const_units'][fct][k]
            output['fixed_const_tokens'].append(
                {
                    'token_name': fct,
                    'token_type': 'fixed_const',
                    'length': 1,
                    'description': dcp_args[fct],
                    'physical_units': fct_physical_units,
                }
            )

        # free const tokens
        output['free_const_tokens'] = []
        for fct in token_args['free_const_tokens']:
            fct_physical_units = {}
            for k in range(len(unit_names)):
                unit_name = unit_names[k]
                fct_physical_units[unit_name] = token_args['free_const_units'][fct][k]
            output['free_const_tokens'].append(
                {
                    'token_name': fct,
                    'token_type': 'free_const',
                    'length': 1,
                    'description': dcp_args[fct],
                    'physical_units': fct_physical_units,
                }
            )

        return self.dict2block(output)

    def describeInitialPrompt(self, token_args=DEFAULT_TOKEN_ARGS, dcp_args=DEFAULT_TOKEN_DESCRIPTION):
        '''
        describe the initial tokens and save them to a md file
        '''
        # prompt = f"## Initial Prompt\n"
        # prompt += f"### Tokens\n"
        prompt = ""
        prompt += self.describeTokens(token_args, dcp_args)
        self.save_prompt(prompt, f"{self.basic_folder}/initial_variables.md")
        return prompt

    def get_new_combination(self):
        self.init_memory()
        # analyze the background
        print("\n[yellow]Analyzing the background...[/yellow]")
        prompt = "Please analyze why the form of the IDM formula and the GHR formula are formed in that way."
        self.talk2llm(prompt, memorize=True, update=True)

        code_blocks = self.extract_python_code_blocks(f"{self.basic_folder}/initial_variables.md")
        self.execute_python_code_blocks(code_blocks)
        print("\n[yellow]Inputing initial infomation...[/yellow]")
        prompt = "Please remember the following table of basic variables, constants, and operators\n" + code_blocks[0]
        self.talk2llm(prompt, memorize=True, update=True)

        operator_meaning_file = f"{self.basic_folder}/operator.md"
        operator_meaning_prompt = self.file2str(operator_meaning_file)
        print("\n[yellow]Informing the rules...[/yellow]")
        prompt = operator_meaning_prompt
        self.talk2llm(prompt, memorize=True, update=True)

        print("\n[yellow]Creating the new combination...[/yellow]")
        prompt = (
            "Please refer to the table and use the symbols and operators to create at least 40 combination expressions with any units. "
            "And explain their physical unit and physical meaning respectively. ")
        self.talk2llm(prompt, memorize=True, update=True)

        # save the memory
        self.save_memory(os.path.join(self.md_save_folder, "new_combinations.md"))

    def init_memory(self, include_background=True):
        if not include_background:
            background_prompt = "The background is not included."
        else:
            background_file = f"{self.basic_folder}/background.md"
            background_prompt = self.file2str(background_file)
        system_message = textwrap.dedent(
            f'''
{delimiter} Background of car-following Models:
{background_prompt}
You are required to act as a helpful assistant, you can provide guidance for constructing car-following models.
Not only consider operator rules, but also unit constraints and physical meanings.
Your response should use the following format:
<reasoning>
<reasoning>
<repeat until you decide to output your final answer>
Response to user:
{delimiter} <Your final output>
Make sure to include {delimiter} to seperate every step.
            ''')
        self.memory = [
            SystemMessage(content=system_message)
        ]

    def talk2llm(self,
                 prompt,
                 memorize=False,
                 update=False):
        '''
        Talk to the LLM
        '''
        # update the memory
        if memorize:
            messages = copy.deepcopy(self.memory)

        else:
            system_message = textwrap.dedent(
                f'''
                You are required to act as a helpful assistant.
                '''
            )
            messages = [SystemMessage(content=system_message)]

        # human_message
        human_message = textwrap.dedent(
            f'''
{delimiter} Human request:\n
{prompt}
            '''
        )
        messages.append(HumanMessage(content=human_message))
        print(prompt)

        # stream out the messages
        response = ""
        for chunk in self.llm.stream(messages):
            response += chunk.content
            print(chunk.content, end="", flush=True)

        # update the memory
        if update:
            self.memory.append(HumanMessage(content=human_message))
            self.memory.append(AIMessage(content=response))

        # save repsonse to a md file
        for i in range(self.max_iter):
            if not os.path.exists(f"{self.md_save_folder}/response_{i}.md"):
                self.save_prompt(response, f"{self.md_save_folder}/response_{i}.md")
                break

        # execute the python code blocks in the response
        code_blocks = self.extract_python_code_blocks(f"{self.md_save_folder}/response_{i}.md")
        self.execute_python_code_blocks(code_blocks)

        # sleep for a while, avoid the rate limit
        print(f"[green]\nSleeping for {self.sleep_time} seconds...\n[/green]")
        time.sleep(self.sleep_time)

        return response

    def extract_python_code_blocks(self, markdown_file):
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

    def execute_python_code_blocks(self, code_blocks):
        '''
        execute python code blocks
        '''
        for code_block in code_blocks:
            try:
                exec(code_block)
            except Exception as e:
                print(f"Error executing code block: {e}")

    def clear_combinations(self, library):
        library["combination_tokens"] = []
        library["combination_description"] = {}
        library["combination_units"] = {}
        library["combination_prefix_expression"] = {}
        return library

    def update_token_args_combinations(self, token_args=None, new_combinations=None):
        # Function to add new combinations to the symbol library
        def add_new_combinations(new_token_args, new_combs):
            for token, details in new_combs.items():
                infix_expression = details["name"]
                if infix_expression not in new_token_args["combination_tokens"]:
                    new_token_args["combination_tokens"].append(infix_expression)
                    new_token_args["combination_description"][infix_expression] = details["description"]
                    new_token_args["combination_units"][infix_expression] = details["units"]
                    new_token_args["combination_prefix_expression"][infix_expression] = details["prefix_expression"]
            return new_token_args

        # Add the new combinations
        new_token_args = copy.deepcopy(token_args)
        add_new_combinations(new_token_args, new_combinations)
        return new_token_args

    def update_token_args_combinations_RAG(self, token_args=None, new_combinations=None):
        # Function to add new combinations to the symbol library
        def add_new_combinations(new_token_args, new_comb):
            infix_expression = new_comb["name"]
            if infix_expression not in new_token_args["combination_tokens"]:
                new_token_args['combination_tokens'].append(infix_expression)
                # new_token_args["combination_infix_expression"].append(infix_expression)
                new_token_args["combination_description"][infix_expression] = new_comb["description"]
                new_token_args["combination_units"][infix_expression] = new_comb["units"][:2]
                new_token_args["combination_prefix_expression"][infix_expression] = new_comb["prefix_expression"]
            return new_token_args

        # Add the new combinations
        new_token_args = copy.deepcopy(token_args)
        for new_combination in new_combinations:
            add_new_combinations(new_token_args, new_combination)
        return new_token_args

    def delete_token_args_combinations_RAG(self, token_args=None, worst_combinations=None):
        # Function to add new combinations to the symbol library
        def delete_new_combinations(new_token_args, worst_combinations):
            for (k, worst_combination_values) in worst_combinations.items():
                # print(worst_combination_values)
                infix_expression = worst_combination_values["name"]
                if infix_expression in new_token_args["combination_tokens"]:
                    new_token_args['combination_tokens'].remove(infix_expression)
                    # new_token_args["combination_infix_expression"].remove(infix_expression)
                    new_token_args["combination_description"].pop(infix_expression)
                    new_token_args["combination_units"].pop(infix_expression)
                    new_token_args["combination_prefix_expression"].pop(infix_expression)
            return new_token_args

        # Add the new combinations
        new_token_args = copy.deepcopy(token_args)
        delete_new_combinations(new_token_args, worst_combinations)
        return new_token_args

    def dict2block(self, dict_obj, block_type='python'):
        dict_str = json.dumps(dict_obj, indent=4)
        markdown_str = f"```{block_type}\n{dict_str}\n```\n"
        return markdown_str

    def file2str(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read() + '\n'

    def get_message_type(self, message):
        '''
        return the message type
        '''
        if type(message) == AIMessage:
            return "AI"
        elif type(message) == HumanMessage:
            return "Human"
        elif type(message) == SystemMessage:
            return "System"
        else:
            raise ValueError(f"Unknown message type '{type(message)}'!")

    def save_memory(self, filename):
        '''
        save the memory to a md file, including the message type
        '''
        with open(filename, 'w', encoding='utf-8') as f:
            pass
        for message in self.memory:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"## message type: {self.get_message_type(message)}\n")
                f.write(f"{message.content}\n")

    def init_tokens_by_llm(self):
        '''
        instruct llm to propose new car-following models and extract valuable tokens
        '''

        self.init_memory()
        # analyze the background
        print("\n[yellow]Analyzing the background...[/yellow]")
        prompt = "Please analyze why the form of the IDM formula and the GHR formula are formed in that way."
        self.talk2llm(prompt, memorize=True, update=True)

        # propose new car-following models
        for i in range(self.initial_trial):
            print(f"\n[yellow]Trial {i}...[/yellow]")
            prompt = "Please try to propose a new car follwing model and explain why your formula takes this form."
            self.talk2llm(prompt, memorize=True, update=True)

        # summarize the tokens
        print("\n[yellow]Summarizing the tokens...[/yellow]")
        prompt = "Based on the new car-following models you propoesed, please provide " + \
                 "some symbols that you think may be included in excellent car-following models. " + \
                 "Exlpain their physical meanings and dimensions." + \
                 "Please note, you should provide at least 10 symbols with formats like this:\n"
        example_dict = {
            "token_name": "delta_v",
            "is_basic": True,
            "token_type": "input_variable",
            "length": 1,
            "description": "speed difference between ego and the fronted",
            "physical_units": {
                "m": 1,
                "s": -1
            },
            "recipes": {
                "tokens": ["delta_v"],
                "operators": [],
                "prefix_expression": ["delta_v"],
            }
        }
        prompt += self.dict2block(example_dict)
        prompt += "There is a python dictionary called 'tokens_info', there are some basic symbols in it: \n"
        prompt += self.file2str(f"{self.basic_folder}/initial_variables.md")
        prompt += "Then, append the new symbols to it, which should be basic symbols with 'is_basic' set to True."

        self.talk2llm(prompt, memorize=True, update=True)

        # save the memory
        self.save_memory(os.path.join(self.md_save_folder, "initial_python_tokens.md"))

    def combine_tokens_by_llm(self, new_tokens_number=3, basic_tokens_file="initial_variables.md",
                              combinations_file="combinations.md"):
        '''
        combine new tokens according to the reward of combinations ouput by SR construction
        '''

        # init memory
        self.init_memory(include_background=False)

        # combine tokens
        print("\n[yellow]Combining new tokens according to their rewards...[/yellow]")
        prompt = "Here are some tokens, the physical meanings and dimensions of these tokens are also provided:\n"
        prompt += self.file2str(f"{self.basic_folder}/{basic_tokens_file}")
        prompt += "Here are also some combinations of these basic tokens, together with their rewards of their fitting effects and complexity." + \
                  "The bigger the reward, the better the combination. However, the output cannot reflect whether these combinations have good physical significance.\n"
        prompt += self.file2str(f"{self.basic_folder}/{combinations_file}")
        prompt += "Then, please combine new tokens according to the rewards of combinations and their physical meanings." + \
                  "The new tokens' format should be like this:\n"
        example_dict = {
            "token_name": "v_ego/v_fronted",
            "token_type": "input_variable",
            "is_basic": False,
            "description": "speed difference between ego and the fronted",
            "physical_units": {
                "m": 0,
                "s": 0
            },
            "recipes": {
                "variable_tokens": ["v_ego", "v_fronted"],
                "operator_tokens": ["divide"],
                "prefix_expression": ["divide", "v_ego", "v_fronted"],
            }
        }
        prompt += self.dict2block(example_dict)
        prompt += f"Then, add {new_tokens_number} new combinations to tokens_info['variable_tokens'], which should not only have excellent fitting effects and low complexity, but also have good physical meanings."
        prompt += "Please complete a python block which can be executed directly to add the new tokens to the tokens_info."

        self.talk2llm(prompt, memorize=True, update=True)

        # save the memory
        self.save_memory(os.path.join(self.md_save_folder, "combine_tokens_by_llm.md"))

    def combine_tokens_by_llm_first_step(self):

        # init memory
        self.init_memory(include_background=False)

        # combine tokens
        prompt = self.dict2block(DEFAULT_TOKEN_ARGS)
        prompt += "The above is the description of the existing tokens in the token library." + \
                  f"Please create at least one new symbol combination for each free constant token, which should be semantically useful for creating a car-following model," + \
                  "and provide the prefix expressions and physical units for the combinations.\n" + \
                  "For example, if you want to add a new combination 'v/v0', you should append its name 'v/v_0' to 'combination_tokens', " + \
                  "and its physical unit to 'combination_units', its description to 'combination_description' ,and its prefix expression to 'combination_prefix_expression'."

        self.talk2llm(prompt, memorize=True, update=True)

        # save the memory
        self.save_memory(os.path.join(self.md_save_folder, "combine_tokens_by_llm_first_step.md"))

    def get_valuable_from_tokens(self):
        '''
        init tokens from idm formula
        '''
        self.init_memory()
        # analyze the background
        print("\n[yellow]Analyzing the background...[/yellow]")
        prompt = "Please analyze why the form of the IDM formula and the GHR formula are formed in that way."
        self.talk2llm(prompt, memorize=True, update=True)

        # tell the LLM the infomation of the initial tokens
        code_blocks = self.extract_python_code_blocks(f"{self.basic_folder}/initial_variables.md")
        self.execute_python_code_blocks(code_blocks)
        print("\n[yellow]Inputing initial infomation...[/yellow]")
        prompt = "Please remember the following table of basic variables, constants, and operators, especially their description.\n" + \
                 self.dict2block(code_blocks[0])
        self.talk2llm(prompt, memorize=True, update=True)

        # inform the rules of the operators, 1.extract 2.sort
        operator_meaning_file = f"{self.basic_folder}/learn_meaning.md"
        operator_meaning_prompt = self.file2str(operator_meaning_file)
        print("\n[yellow]Informing the rules...[/yellow]")
        prompt = operator_meaning_prompt
        self.talk2llm(prompt, memorize=True, update=True)

        # extract and sort the tokens from idm formula
        print("\n[yellow]Extracting and sorting...[/yellow]")
        expression = "$a = alpha * (1 - (v/v0)^2 - (s0/s)^2)$"
        # expression="$a_{IDM}=\alpha\times[1-(\frac{v}{v_0})^2-(\frac{s_0}{s})^2]$"
        prompt = (
                "Please use two steps for this formula in detail and provide a detailed explanation of the reason: " + expression)
        self.talk2llm(prompt, memorize=True, update=True)

        # save the memory
        self.save_memory(os.path.join(self.md_save_folder, "memory.md"))

    def extract_valuable(self, new_tokens_number=3, origin_token_dict=None,
                         trainResultFile="different_expressions_evolution0.md",
                         evolution=0):
        # init memory
        self.init_memory(include_background=False)

        # combine tokens
        print("\n[yellow]Combining new tokens according to their rewards...[/yellow]")
        prompt = "This is the orginal symbol library for the car following model. The forms, physical units and physical descriptions of these tokens or combinations are provided: \n"
        prompt += self.dict2block(origin_token_dict)

        prompt += "We input a set of x-y data and fit some formula expressions for the car following algorithm based on these symbols. " + \
                  "Among them, rewards are all within the range of [0,1], and the closer they are to 1, the better the performance of the formula in this aspect.\n"
        prompt += "Fit_accuracy_reward is the most important, indicating the degree of fit between the formula and the data points; " + \
                  "Fit_similarity_reward is the second most important factor, indicating the degree of similarity between the formula and the form we expect; " + \
                  "Fit_complete_reward is the least important, as a larger value indicates a lower complexity of the formula. " + \
                  "Sum_reward is the weighted sum of three types of rewards.\n"
        prompt += self.file2str(f"{self.trainResultFolder}{trainResultFile}")

        prompt += f"Please extract at least {new_tokens_number} new combinations of tokens from the above programs. (n2 means square, for example, n2(x) means $x^2$; inv means inverse, for example, inv(x) means $x^\{-1}$)" + \
                  "When extracting new combination of tokens, please refer to the following rules and make sure the combinations do not : \n"
        prompt += self.file2str(f"{self.basic_folder}/idm_logic.md")
        prompt += "Please (a) output the new token combinations which are not in orginal symbol library and (b) explain one by one why they comply with these four rules, "
        prompt += "(c) describe which expression you extracted it from, and (d) point out which original tokens and original combinations you used to combine a new one"
        prompt += "(You're encouraged to use old combinations to combine new combinations, which can refloect the process of your incremental learning, "
        prompt += "if you think there are no combinations worth extracting from the above expressions, you can also directly generate combinations that you think are meaningful based on the above four rules). \n"
        prompt += "Your output should contain the reasoning of (a)(b)(c)(d), and we encourage you to provide "

        prompt += "Then, return a python block containing a dict of new combinations. " + \
                  "Here is an example:\n"
        prompt += self.file2str(os.path.join(self.basic_folder, "new_combination_form.md"))
        prompt += "The format of your output should be like this, a python block containing a dict of combinations should be only output at the final step. :\n"
        prompt += "New_combination_1 : ...(infix expression and why it complies with the rules)\n"
        prompt += "New_combination_2 : ...(infix expression and why it complies with the rules)\n"
        prompt += "...\n"
        prompt += "Python block: \n"
        prompt += "(a python block with format same as the example above)\n"

        response = self.talk2llm(prompt, memorize=True, update=True)
        self.save_memory(os.path.join(self.trainResultFolder, f"extract_valuable{evolution}.md"))

        # examine the combinations
        print("\n[yellow]Combination is done, examine the new combinations...[/yellow]")
        self.init_memory(include_background=False)
        code_blocks = self.extract_python_code_blocks(
            os.path.join(self.trainResultFolder, f"extract_valuable{evolution}.md"))
        new_combinations = eval(code_blocks[-1])
        new_combinations_str = self.dict2block(new_combinations)

        prompt = "First, I will provide you with the orginal symbol library for the car following model, The physical descriptions, physical units and prefix expressions of them are provided.\n"
        prompt += self.dict2block(origin_token_dict)
        prompt += "I will provide you with a python dictionary of token combinations for constructing a car-following model." + \
                  "The physical descriptions, physical units and prefix expressions of them are provided. Here it is:\n"
        prompt += new_combinations_str
        prompt += "please check the dictionary with following steps, then output a improved python block which has a same format with the old one.\n"
        prompt += "Improving Steps:\n"
        prompt += "1. Check if each combination has appeared in the original library by comparing the prefix expression of the new combination to whether it has the same situation as the original library.\n"
        prompt += "If the new combination has appeared in the original library, delete this combination. \n"
        prompt += "2. Check if each combination in the dictionary is legal, specifically, check if the infix expression, prefix expression and physical units "
        prompt += "(physical units should be represented by a two-dimensional list, where the first dimension represents meters and the second dimension represents seconds) "
        prompt += "of each combination correspond correctly. If the combination is illegal, replace it with a valid combination.\n"
        prompt += "3. Check if the formula form of each combination corresponds correctly to its physical meaning, you can refer to the original symbol library to check the meaning of each token.\n"
        prompt += "If the formula form of the combination does not correspond correctly to its physical meaning, replace its 'description' that better fits its physical meaning.\n"
        prompt += "4. For each combination, evaluate whether it can be used to combine a car following model that meets the following requirements,\n"
        prompt += self.file2str(f"{self.basic_folder}/idm_logic.md")
        prompt += "If the combination does not meet the requirements, delete it.\n"
        prompt += "5. Check if the symbol combination is too complex. We believe that it is unreasonable to combine overly complex expressions based on the symbol library at once. "
        prompt += "However, if the combination in the library is a sub expression of the current combination (for example, if the prefix expression is [div, v, v0] and is a sub "
        prompt += "expression of [n2, div, v, v0]), we can tolerate the complexity of this combination. Please evaluate the complexity of these symbol combinations. "
        prompt += "If they are too complex, please extract a sub expression from these symbol combinations (of course, this expression also needs to meet the conditions mentioned earlier) "
        prompt += "as a substitute for this symbol combination.\n"
        prompt += "6. Finally, Optimize the names and physical descriptions of the preserved combinations to make them easier for humans to understand. "
        prompt += "For example, for combinations with dimensions of velocity, simplify the combination name to 'v_ {...}', where Represents a more condensed physical meaning of this combination. "
        prompt += "For example, for symbols whose physical meaning only describes their calculation methods, abstract their more concise physical meaning and replace the original description "
        prompt += "(for example, 'v/v0' can be described as a speed related influencing factor).\n"
        prompt += "For each combiination, please improve or delete it following the above steps, and a python block containing a dict of new combinations should be only output at the final step. "

        response = self.talk2llm(prompt, memorize=True, update=True)
        self.save_memory(os.path.join(self.trainResultFolder, f"extract_valuable{evolution}.md"))

    def select_best_combinations(self, origin_token_dict, top_k, trainResultFile, evolution):
        # init memory
        self.init_memory(include_background=False)
        print(f"\n[yellow]Selecting the best {top_k} combinations...[/yellow]")

        prompt = "This is the orginal symbol library for the car following model. The forms, physical units and physical descriptions of these tokens are provided: \n"
        prompt += self.dict2block(origin_token_dict)

        prompt += "We input a set of x-y data and get some elite expressions based on the tokens and combinations above, these expressions have outstanding performance to fit the car-following trajectoris." + \
                  "Among them, rewards are all within the range of [0,1], and the closer they are to 1, the better the performance of the formula in this aspect.\n"
        prompt += "Fit_accuracy_reward is the most important, indicating the degree of fit between the formula and the data points; " + \
                  "Fit_similarity_reward is the second most important factor, indicating the degree of similarity between the formula and the form we expect; " + \
                  "Fit_complete_reward is the least important, as a larger value indicates a lower complexity of the formula. " + \
                  "Sum_reward is the weighted sum of three types of rewards.\n"
        prompt += self.file2str(f"{self.trainResultFolder}{trainResultFile}")

        prompt += f"Now, we need to reduce the number of combinations in the symbol library to {top_k}, to reduce the search space of reinforcement learning agents. "
        prompt += "Please follow the following steps to simplify the symbol library:\n"
        prompt += "1. Observe which combinations in the existing symbol library appear less frequently in the above elite expressions. "
        prompt += "The lower the frequency of symbol combinations, the more they should be considered for deletion.\n"
        prompt += "2. Check which combinations are less suitable for the following rules, and the loss suitable it is to fit the following rules, "
        prompt += "the more they should be considered for deletion. \n"
        prompt += "3. Check which combinations have poor interpretability or conciseness, and more complex and difficult to understand combinations should be considered for deletion.\n"
        prompt += f"4. Finally, output a python block containing a dict of the best {top_k} combinations. "
        prompt += "Here is an example and the python dict you output should have the same format:\n"
        prompt += self.file2str(os.path.join(self.basic_folder, "new_combination_form.md"))
        prompt += "Please reasoning step by step, and a python block containing a dict of combinations should be only output at the final step. "

        response = self.talk2llm(prompt, memorize=True, update=True)
        self.save_memory(os.path.join(self.trainResultFolder, f"updated_combinations{evolution}.md"))

    def save_prompt(self, prompt, filename):
        '''
        save str prompt to a file
        '''
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(prompt)
        return

    def add_prompt_old(self, prompt, filename):
        '''
        save str prompt to a file
        '''
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(prompt)
        return

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

    def record_trainResult_history(self, trainResult, writePath):
        '''
        Record reward history of trainResult
        '''
        output = {}
        for i in range(len(trainResult["different_candidates_history"]) - 1, -1, -1):
            output['epoches_history'] = list(range(len(trainResult["mean_R_train_history"])))
            output['mean_R_train_history'] = trainResult[
                "mean_R_train_history"]  # mean of R_train Rewards in each epoch
            output['mean_R_history'] = trainResult["mean_R_history"]  # mean of Rewards in each epoch
            output['complexity_reward_history'] = list(np.array(trainResult["sub_rewards_history"])[:, 0])
            output['similarity_reward_history'] = list(np.array(trainResult["sub_rewards_history"])[:, 1])
            output['rmse_reward_history'] = list(np.array(trainResult["sub_rewards_history"])[:, 2])
            output['mean_complexity_reward_history'] = list(np.array(trainResult["mean_R_train_sub_history"])[:, 0])
            output['mean_similarity_reward_history'] = list(np.array(trainResult["mean_R_train_sub_history"])[:, 1])
            output['mean_rmse_reward_history'] = list(np.array(trainResult["mean_R_train_sub_history"])[:, 2])
            output['best_complexity_reward_history'] = list(np.array(trainResult["similarity_best_R_sub_history"])[:, 0])
            output['best_similarity_reward_history'] = list(np.array(trainResult["similarity_best_R_sub_history"])[:, 1])
            output['best_rmse_reward_history'] = list(np.array(trainResult["similarity_best_R_sub_history"])[:, 2])
            output['rewards_history'] = trainResult["rewards_history"]  # epoch best reward
            output['best_R_history'] = trainResult["best_R_history"]  # evolution best reward history
        prompt = "## Train result of reward history\n"
        prompt += self.dict2block(output)
        self.save_prompt(prompt, writePath)

    def write_evolution_expressions(self, trainResult, writePath, save_number=20):
        '''
        Describe expressions of trainResult
        '''
        output = {}
        R_argsort = np.argsort(trainResult["different_rewards_history"])[::-1][:save_number]
        for i in range(0, len(R_argsort)):  # sum_reward from large to small
            # sum and rmse
            if trainResult['different_sub_rewards_history'][R_argsort[i]][2] > 0.0:
                output['expression' + str(i)] = {}
                output['expression' + str(i)]['prefix_expression'] = trainResult[
                    "different_candidates_history"][R_argsort[i]].get_prefix_expression()
                output['expression' + str(i)]['infix_expression'] = trainResult[
                    "different_candidates_infix_history"][R_argsort[i]]
                # output['expression' + str(i)]['infix_expression'] = str(trainResult["different_candidates_history"][
                #     R_argsort[i]].get_infix_sympy(
                #     do_simplify=True))
                # output['expression' + str(i)]['print_expression'] = trainResult["candidates_history"][
                #     R_argsort[i]].get_print_expression(do_simplify=True)

                output['expression' + str(i)]['fit_accuracy_reward'] = \
                    trainResult["different_sub_rewards_history"][R_argsort[i]][
                        2]
                output['expression' + str(i)]['fit_similarity_reward'] = \
                    trainResult["different_sub_rewards_history"][R_argsort[i]][1]
                output['expression' + str(i)]['fit_complexity_reward'] = \
                    trainResult["different_sub_rewards_history"][R_argsort[i]][0]
                output['expression' + str(i)]['sum_reward'] = trainResult["different_rewards_history"][R_argsort[i]]

        prompt = "## Train result\n"
        prompt += self.dict2block(output)
        self.save_prompt(prompt, writePath)

    def extract_valuable_old(self, new_tokens_number=3, origin_token_dict=None,
                             trainResultFile="different_expressions_evolution0.md",
                             evolution=0):
        # init memory
        self.init_memory(include_background=False)

        # combine tokens
        print("\n[yellow]Combining new tokens according to their rewards...[/yellow]")
        prompt = "This is the symbol library for the car following model. The forms, physical units and physical descriptions of these tokens are provided:\n"
        prompt += "Among them, it is important to focus on combinations, which are valuable symbol combinations. " + \
                  "The infix expressions, physical descriptions, physical units and prefix expressions of them are provided.\n"
        prompt += self.dict2block(origin_token_dict)

        prompt += "We input a set of x-y data and fit some formula expressions for the car following algorithm based on these symbols. " + \
                  "Among them, rewards are all within the range of [0,1], and the closer they are to 1, the better the performance of the formula in this aspect.\n"
        prompt += "Fit_accuracy_reward is the most important, indicating the degree of fit between the formula and the data points; " + \
                  "Fit_similarity_reward is the second most important factor, indicating the degree of similarity between the formula and the form we expect; " + \
                  "Fit_complete_reward is the least important, as a larger value indicates a lower complexity of the formula. " + \
                  "Sum_reward is the weighted sum of three types of rewards.\n"
        prompt += self.file2str(f"{self.trainResultFolder}{trainResultFile}")

        prompt += f"Please extract at least {new_tokens_number} of the most important new combinations from these formulas. " + \
                  "1. These new combinations need to be different from the combinations in the symbol library. " + \
                  "2. New combinations should not only have high fit accuracy, high similarity and low complexity, but also have good physical meanings. " + \
                  "3. Please note that although the expressions we provide have good rewards, the positions where the free constants appear in them may not match their original physical meanings. " + \
                  "For example, the free constants that originally modify the effect of following speed may appear at the positions where the following distance is modified. " + \
                  "Therefore, please pay attention to this when extracting new combinations and replace the free constant symbols in the combination with symbols in the symbol library that match the physical description" + \
                  "4. The prefix expression length of these combinations needs to be less than or equal to 13.\n"

        prompt += "Please only return a python block containing a dict of new combinations. This block can be executed directly." + \
                  "Here is an example:\n"
        prompt += self.file2str(os.path.join(self.basic_folder, "new_combination_form.md"))

        self.talk2llm(prompt, memorize=True, update=True)
        self.save_memory(os.path.join(self.trainResultFolder, f"extract_valuable{evolution}.md"))

    def single_step_increase(self, new_tokens_number=3, origin_token_dict=None,
                             trainResultFile="different_expressions_evolution0.md",
                             evolution=0):
        # init memory
        self.init_memory(include_background=False)

        # combine tokens
        print("\n[yellow]Combining new tokens according to their rewards...[/yellow]")
        prompt = "This is the symbol library for the car following model. The forms, physical units and physical descriptions of these tokens are provided:\n"
        prompt += "Among them, it is important to focus on combinations, which are valuable symbol combinations. " + \
                  "The infix expressions, physical descriptions, physical units and prefix expressions of them are provided.\n"
        prompt += self.dict2block(origin_token_dict)

        prompt += "We input a set of x-y data and fit some formula expressions for the car following algorithm based on these symbols. " + \
                  "Among them, rewards are all within the range of [0,1], and the closer they are to 1, the better the performance of the formula in this aspect.\n"
        prompt += "Fit_accuracy_reward is the most important, indicating the degree of fit between the formula and the data points; " + \
                  "Fit_similarity_reward is the second most important factor, indicating the degree of similarity between the formula and the form we expect; " + \
                  "Fit_complete_reward is the least important, as a larger value indicates a lower complexity of the formula. " + \
                  "Sum_reward is the weighted sum of three types of rewards.\n"
        prompt += self.file2str(f"{self.trainResultFolder}{trainResultFile}")

        prompt += f"Please extract at least {new_tokens_number} of the most important new combinations from these formulas. " + \
                  "1. These new combinations need to be different from the existing combinations in the symbol library. " + \
                  "2. New combinations should not only have high fit accuracy, high similarity and low complexity, but also have good physical meanings. " + \
                  "3. In addition, the prefix expression length of these combinations needs to be less than or equal to 13. " + \
                  "4. The resulting combination expression needs to be tokens from the symbol library, which only go through a single step combination with an operator, " + \
                  "such as T*v and s combined with a division sign /, to obtain T * v / s. " + \
                  "5. In the process of single step combination, it is best for the elements of two combinations to contain at least one existing combination, such as v/v0, etc.\n"

        prompt += "Please only return a python block containing a dict of new combinations. This block can be executed directly." + \
                  "Here is an example:\n"
        prompt += self.file2str(os.path.join(self.basic_folder, "new_combination_form.md"))

        self.talk2llm(prompt, memorize=True, update=True)
        self.save_memory(os.path.join(self.trainResultFolder, f"single_step_increase{evolution}.md"))

    def get_semantic_score_md(self, combinations, evolution=0, have_get_semantic_score=False):
        self.init_memory(include_background=False)  # init memory
        print(f"\n[yellow]Get the semantic score of combinations...[/yellow]")
        combinations_partial_info = {}
        for combination_name in combinations.keys():
            combinations_partial_info[combination_name] = {}
            combinations_partial_info[combination_name]["combination_infix_expression"] = \
                combinations[combination_name]["combination_infix_expression"]
            combinations_partial_info[combination_name]["combination_description"] = combinations[combination_name][
                "combination_description"]
        combinations_str = self.dict2block(combinations)

        def check_answer():
            code_blocks = self.extract_python_code_blocks(
                f"{self.trainResultFolder}semantic_score_dict{evolution}.md")
            semantic_score_dict = eval(code_blocks[-1])
            if combinations.keys() - semantic_score_dict.keys() == {} and semantic_score_dict.keys() - combinations.keys() == {}:
                print("The semantic score of combinations is successfully obtained.")
            else:
                raise ValueError

        try:
            # prompt = "First, I will provide you with the orginal symbol library for the car following model, The physical descriptions, physical units and prefix expressions of them are provided.\n"
            # prompt += self.dict2block(origin_token_dict)
            prompt = "I will provide you with a python dictionary of token combinations for constructing a car-following model." + \
                     "The physical descriptions, physical units and prefix expressions of them are provided. Here it is:\n"
            prompt += combinations_str
            # self.talk2llm(prompt, memorize=True, update=True) # can not talk twice
            #
            # self.init_memory(include_background=False)  # clear the memory
            prompt += "Please check the dictionary with following steps, then output each combination and its corresponding semantic score, " + \
                      "which is a score from 0 to 1 for whether the combination conforms to the physical semantics required by the vehicle follower, with higher scores representing greater conformity.\n"
            prompt += "Improving Steps:\n"
            prompt += "1. If there are multiple combinations with the same physical meaning and form, the second and subsequent combinations should be scored low.\n"
            prompt += "2. Check if each combination in the dictionary is legal, specifically, check the infix expression, prefix expression and physical units. " + \
                      "If the combination is illegal, its semantic score should be 0.\n"
            prompt += "3. Check which combinations are less interpretable or concise. The more complex and difficult to understand one combination, the less its semantic score.\n"
            prompt += "4. For each combination, evaluate whether it can be used to combine a car following model that meets the following requirements,\n"
            prompt += self.file2str(f"{self.basic_folder}/idm_logic.md")

            prompt += "You need to make every combination in the library obtain a semantic score. "
            prompt += "Here is an example, the scores are for reference only. " + \
                      "The python dict you output should have the same format:\n"
            if have_get_semantic_score == False:
                prompt += self.file2str(os.path.join(self.basic_folder, "combination_semantic_score.md"))
            else:
                code_blocks = self.extract_python_code_blocks(
                    f"{self.trainResultFolder}semantic_score_dict{evolution - 1}.md")
                combinations_sematic_score = code_blocks[-1]
                prompt += combinations_sematic_score

            # prompt += "Please reasoning step by step, and a python block containing a dict of combinations should be only output at the final step. "
            self.talk2llm(prompt, memorize=True, update=True)
            self.save_memory(filename=f"{self.trainResultFolder}semantic_score_dict{evolution}.md")
            check_answer()

        except ValueError:  # make result right form
            # check_message = textwrap.dedent(f"""
            #
            #                         """)
            #
            #     messages = [
            #         HumanMessage(content=check_message),
            #     ]
            #     with get_openai_callback() as cb:
            #         check_response = self.llm(messages)
            prompt = "Your output result is incorrect. You need to make every combination in the library obtain a semantic score. " + \
                     "The key of the dictionary you output should be the same as the key of the combination dictionary.\n"
            prompt += "I will provide you with a python dictionary of combinations for constructing a car-following model." + \
                      "The physical descriptions, physical units and prefix expressions of them are provided. Here it is:\n"
            prompt += combinations_str
            prompt += "Here is an example, the scores are for reference only. " + \
                      "Please output each combination and its corresponding semantic score in the same format:\n"
            if have_get_semantic_score == False:
                prompt += self.file2str(os.path.join(self.basic_folder, "combination_semantic_score.md"))
            else:
                code_blocks = self.extract_python_code_blocks(
                    f"{self.trainResultFolder}semantic_score_dict{evolution - 1}.md")
                combinations_sematic_score = code_blocks[-1]
                prompt += combinations_sematic_score
            self.talk2llm(prompt, memorize=True, update=True)
            self.save_memory(filename=f"{self.trainResultFolder}semantic_score_dict{evolution}.md")
            check_answer()

    # # make result right form
    # try:
    #     result = int(decision)
    #     if result < 0 or result > 4:
    #         raise ValueError
    # except ValueError:
    #     print("Output is not a int number, checking the output...")
    #     check_message = textwrap.dedent(f"""
    #                 You are a output checking assistant who is responsible for checking the output of another agent.
    #
    #                 The output you received is: {decision}
    #
    #                 Your should just output the right int type of action_id, with no other symbols or delimiters.
    #                 i.e. :
    #                 | Action_id | Action Description                                     |
    #                 |--------|--------------------------------------------------------|
    #                 | 0      | Turn-left: change lane to the left of the current lane |
    #                 | 1      | IDLE: remain in the current lane with current speed   |
    #                 | 2      | Turn-right: change lane to the right of the current lane|
    #                 | 3      | Acceleration: accelerate the vehicle                 |
    #                 | 4      | Deceleration: decelerate the vehicle                 |
    #
    #                 You answer format would be:
    #                 {delimiter} <correct action_id within 0-4>
    #
    #                 For example, if you decide to decelerate, then output:
    #                 {delimiter} '4'
    #                 """)
    #
    #     messages = [
    #         HumanMessage(content=check_message),
    #     ]
    #     with get_openai_callback() as cb:
    #         check_response = self.llm(messages)
    #     result = int(check_response.content.split(delimiter)[-1])

    def write_all_evolutions_best_expressions(self, bestInEvolutions, writePath):
        '''
        Describe expressions of bestInEvolutions
        '''
        output = {}
        for i in range(len(bestInEvolutions["best_program_history"])):
            output['expression' + str(i)] = {}
            output['expression' + str(i)]['prefix_expression'] = bestInEvolutions["best_program_history"][
                i].get_prefix_expression()
            output['expression' + str(i)]['infix_expression'] = str(
                bestInEvolutions["best_program_history"][i].get_infix_sympy(
                    do_simplify=True if len(
                        bestInEvolutions["best_program_history"][i].tokens) < simplify_True_length_limit else False))
            # if len(bestInEvolutions["best_program_history"][i].tokens) >= simplify_True_length_limit:
            #     print(
            #         f"The prefix expression is {len(output['expression' + str(i)]['prefix_expression'])}, too long don't simplify.")
            output['expression' + str(i)]['x_reward'] = bestInEvolutions["best_sub_R_history"][i][3]
            output['expression' + str(i)]['v_reward'] = bestInEvolutions["best_sub_R_history"][i][4]
            output['expression' + str(i)]['fit_accuracy_reward'] = bestInEvolutions["best_sub_R_history"][i][2]
            output['expression' + str(i)]['fit_similarity_reward'] = bestInEvolutions["best_sub_R_history"][i][1]
            output['expression' + str(i)]['fit_complexity_reward'] = bestInEvolutions["best_sub_R_history"][i][0]
            output['expression' + str(i)]['sum_reward'] = bestInEvolutions["best_R_history"][i]
        prompt = "## Train result\n"
        prompt += self.dict2block(output)
        self.save_prompt(prompt, writePath)

    def explain_expressions(self, origin_token_dict=None, bestResultFile='best_result_of_evolutions.md'):
        # init memory
        self.init_memory(include_background=False)

        # combine tokens
        print("\n[yellow]Explain most valuable expressions...[/yellow]")
        prompt = "This is the symbol library for the car following model. The forms, physical units and physical descriptions of these tokens are provided:\n"
        prompt += "Among them, it is important to focus on combinations, which are valuable symbol combinations. " + \
                  "The infix expressions, physical descriptions, physical units and prefix expressions of them are provided.\n"
        prompt += self.dict2block(origin_token_dict)

        prompt += "We have obtained some car following formulas constructed from symbols in the symbol library, representing the acceleration of the following vehicle. " + \
                  "Among them, rewards are all within the range of [0,1], and the closer they are to 1, the better the performance of the formula in this aspect.\n"
        prompt += "Fit_accuracy_reward is the most important, indicating the degree of fit between the formula and the data points; " + \
                  "Fit_similarity_reward is the second most important factor, indicating the degree of similarity between the formula and the form we expect; " + \
                  "Fit_complete_reward is the least important, as a larger value indicates a lower complexity of the formula. " + \
                  "Sum_reward is the weighted sum of three types of rewards.\n"
        prompt += "Please analyze the physical meanings of each part of the following formulas to discover why they have good car following effects.\n"
        prompt += self.file2str(f"{self.trainResultFolder}{bestResultFile}")

        # prompt += "Finally, please add the infix expressions, physical descriptions, physical units and prefix expressions of newly extracted combination to corresponding combination section in the symbol library. "
        # # prompt += "Please complete a python block which can be executed directly to add the new tokens to the symbol library."
        # prompt += "Please only return a python block containing a dict of new combinations. " + \
        #           "Here is an example:\n"
        # prompt += self.file2str(os.path.join(self.basic_folder, "expression_expression_form.md"))

        self.talk2llm(prompt, memorize=True, update=True)
        self.save_memory(os.path.join(self.trainResultFolder, "explain_expressions.md"))

    def explain_new_model(self, origin_token_dict=None, best_result_dict=None):
        # init memory
        self.init_memory(include_background=False)

        # combine tokens
        print("\n[yellow]Explain most valuable expressions...[/yellow]")
        prompt = "This is the symbol library for the car following model. The forms, physical units and physical descriptions of these tokens are provided:\n"
        prompt += "Among them, it is important to focus on combinations, which are valuable symbol combinations. " + \
                  "The infix expressions, physical descriptions, physical units and prefix expressions of them are provided.\n"
        prompt += self.dict2block(origin_token_dict)

        prompt += "We have obtained a car following model constructed from symbols in the symbol library, representing the acceleration of the following vehicle. " + \
                  "Among them, rewards are all within the range of [0,1], and the closer they are to 1, the better the performance of the formula in this aspect.\n"
        prompt += "Fit_accuracy_reward is the most important, indicating the degree of fit between the formula and the data points; " + \
                  "Fit_similarity_reward is the second most important factor, indicating the degree of similarity between the formula and the form we expect; " + \
                  "Fit_complete_reward is the least important, as a larger value indicates a lower complexity of the formula. " + \
                  "Sum_reward is the weighted sum of three types of rewards.\n"
        prompt += "Please analyze the physical meanings of each part of the following model to discover why they have good car following effects.\n"
        prompt += self.dict2block(best_result_dict)

        # prompt += "Finally, please add the infix expressions, physical descriptions, physical units and prefix expressions of newly extracted combination to corresponding combination section in the symbol library. "
        # # prompt += "Please complete a python block which can be executed directly to add the new tokens to the symbol library."
        # prompt += "Please only return a python block containing a dict of new combinations. " + \
        #           "Here is an example:\n"
        # prompt += self.file2str(os.path.join(self.basic_folder, "expression_expression_form.md"))

        self.talk2llm(prompt, memorize=True, update=True)
        self.save_memory(os.path.join(self.trainResultFolder, "explain_expressions.md"))


if __name__ == "__main__":
    print("Agent Start")
    # agent = RAG_AGENT()
    # agent.combine_tokens_by_llm()
    # agent.combine_tokens_by_llm_first_step()
    # agent.talk2llm(prompt="hello",
    #                memorize=False,
    #                update=False)
    # agent.generate_new_symbol("A symbol that represents the acceleration of the following vehicle.")
