import codes.trafficSR.A_sampling.env_generator.SRenv as SRenv
import numpy as np

DEFAULT_REWARD_DICT = {
    "complexity": {
        "weight": 0.5,
        "reward": 0.0
    },
    "similarity": {
        "weight": 0.3,
        "reward": 0.0
    },
    "accuracy": {
        "weight": 0.2,
        "reward": 0.0
    },
    "sum_reward": 0.0
}


class PrefixPath():
    def __init__(
            self,
            token_idx,
            reward_dict,
            free_const_values,
            semi_free_const_values,
    ):
        self.token_idx = token_idx
        self.reward_dict = reward_dict
        self.free_const_values = free_const_values
        self.semi_free_const_values = semi_free_const_values
        self.count = 0
        self.is_reflected = False


class CompressedPrefixTree():
    def __init__(
            self,
    ):
        self.db = {}
        self.token_names_lib = []
        self.N_table = {}
        self.tokens_score_dict = {}

    def reset_names_lib(self, env):
        new_token_names_list = list(env.library.all_token_info_table.token_name_table)
        # if len(self.token_names_lib) != 0:
        #     assert self.token_names_lib[:-1] == new_token_names_list[:len(self.token_names_lib) - 1], \
        #         "The old token names list must be a subset of the new one."
        self.token_names_lib = new_token_names_list

    def reset_N_table(self):
        self.N_table = {}

    def prefix_sequence_to_key(self, prefix_sequence):
        '''
        将前缀序列转换为key
        '''
        key = ""
        for token_idx in list(prefix_sequence):
            if str(token_idx) == "end":
                break
            key += f"{str(token_idx)},"  # 以逗号分隔的token字符串
        return key[:-1]

    def __getitem__(self, prefix_sequence):
        '''
        检索功能,返回前缀树中所有以prefix_sequence为前缀的公式
        '''
        key = self.prefix_sequence_to_key(prefix_sequence)
        return self.db[key]

    def N_prefix_with_X(self, X_sequence):
        '''
        返回前缀树中以X_sequence为前缀的公式的数量
        '''
        # return 0
        key = self.prefix_sequence_to_key(X_sequence)
        N = 0
        for k in self.db.keys():
            if k.startswith(key):
                N += self.db[k].count
        return N

    def __setitem__(self, prefix_sequence, sequence_info):
        '''
        插入功能,将一个完整表达式插入到前缀树中
        插入的表达式是化简的吗？还是原始的？
        存储化简前的表达式，增量式学习的新字符是前缀路径中的一个片段
        '''
        # key = self.prefix_sequence_to_key(prefix_sequence)
        if prefix_sequence not in self.db.keys():
            self.db[prefix_sequence] = PrefixPath(
                token_idx=sequence_info['tokens_idx'],
                reward_dict=sequence_info["reward_dict"],
                free_const_values=sequence_info["free_const_values"],
                semi_free_const_values=sequence_info["semi_free_const_values"],
            )
            self.db[prefix_sequence].count = 1
        else:
            self.db[prefix_sequence].count += 1

    def update_batch_per_epoch(self, env):
        batch_size = env.programs.batch_size
        valid_reward_idx = np.where(env.programs.rewards > 0)
        for batch_idx in valid_reward_idx[0]:
            tokens_idx = list(env.library.tokens_idx[batch_idx])
            # prefix_sequence = [self.token_names_lib[token_idx] for token_idx in tokens_idx]
            prefix_sequence = env.library.prefix_str[batch_idx]
            sequence_info = {}
            sequence_info['tokens_idx'] = tokens_idx
            sequence_info["reward_dict"] = {
                "complexity": {
                    "weight": env.programs.reward_weight['factor_complexity'],
                    "reward": env.programs.sub_rewards[batch_idx][0]
                },
                "similarity": {
                    "weight": env.programs.reward_weight['factor_similarity'],
                    "reward": env.programs.sub_rewards[batch_idx][1]
                },
                "rmse": {
                    "weight": env.programs.reward_weight['factor_rmse'],
                    "reward": env.programs.sub_rewards[batch_idx][2]
                },
                "xrmse": {
                    "weight": 0,
                    "reward": env.programs.sub_rewards[batch_idx][3]
                },
                "vrmse": {
                    "weight": 0,
                    "reward": env.programs.sub_rewards[batch_idx][4]
                },
                "sum_reward": env.programs.rewards[batch_idx],
                "sub_rewards": env.programs.sub_rewards[batch_idx],
            }
            sequence_info["free_const_values"] = env.programs.free_const_values.values[batch_idx]
            sequence_info["semi_free_const_values"] = env.programs.semi_free_const_values.values[batch_idx]
            self.__setitem__(prefix_sequence, sequence_info)

    def update_N_table(self, prefix_str, new_str):
        if not prefix_str in self.N_table.keys():
            self.N_table[prefix_str] = {}
            self.N_table[prefix_str][new_str] = 1
        elif not new_str in self.N_table[prefix_str].keys():
            self.N_table[prefix_str][new_str] = 1
        else:
            self.N_table[prefix_str][new_str] += 1

    def update_batch_per_timestep(self, observation, action, valid_length):
        for b_id in range(observation.shape[0]):
            prefix_str = ''
            vl = int(valid_length[b_id])
            for i in range(vl - 1):
                prefix_str += self.token_names_lib[int(observation[b_id][i])] + ","
            prefix_str = prefix_str[:-1]
            new_str = self.token_names_lib[int(action[b_id])]
            self.update_N_table(prefix_str, new_str)

    def N_of_edge(self, prefix_str, new_str):
        if not prefix_str in self.N_table:
            return 0
        if not new_str in self.N_table[prefix_str]:
            return 0
        return self.N_table[prefix_str][new_str]

    def N_of_edge_batch(self, observation, valid_length):
        res = []
        for b_id in range(observation.shape[0]):  # jiesheng
            res_per_batch = []
            prefix_str = ''
            vl = int(valid_length[b_id])
            for i in range(vl):  # jiesheng
                prefix_str += self.token_names_lib[int(observation[b_id][i])] + ","  # prefix_str of expressions
            for action in range(len(self.token_names_lib) - 1):  # jiesheng
                new_str = self.token_names_lib[int(action)]
                res_per_batch.append(self.N_of_edge(prefix_str, new_str))
            res_per_batch.append(0)  # for end
            res.append(res_per_batch)
        return res

    '''now use'''

    def update_N_table_every_timestep(self, library, new_tokens_idx):
        mask_not_end = new_tokens_idx.numpy() != (len(self.token_names_lib) - 1)
        new_str_numpy = np.full(library.batch_size, fill_value="", dtype='<U32')
        new_str_numpy[mask_not_end] = library.all_token_info_table.token_name_table[new_tokens_idx[mask_not_end]]
        for index in np.where(mask_not_end)[0]:
            prefix_str = library.prefix_str[index]
            new_str = new_str_numpy[index]
            self.update_N_table(prefix_str, new_str)

    def N_of_edge_batch_new(self, library):
        res = np.zeros((library.batch_size, len(self.token_names_lib)))
        for i in range(library.batch_size):
            prefix_str = library.prefix_str[i]
            if prefix_str in self.N_table.keys():
                for key in self.N_table[prefix_str].keys():
                    if key in self.token_names_lib:  # because we delete combinations
                        j = self.token_names_lib.index(key)  # action_idx
                        res[i][j] = self.N_table[prefix_str][key]
        return res

    def update_tokens_score(self, print_score=False, baseline=0.0):
        for token_idx in range(len(self.token_names_lib) - 1):
            count = 0
            sum_score = 0
            for key, info in self.db.items():
                tokens_idx = info.token_idx
                if token_idx in tokens_idx and info.reward_dict["sum_reward"] > baseline:
                    count += info.count
                    sum_score += info.reward_dict["sum_reward"] * info.count
                    self.tokens_score_dict[self.token_names_lib[token_idx]] = sum_score / count if count != 0 else 0

        if print_score:
            print("Tokens score:")
            for token_name, score in self.tokens_score_dict.items():
                print(f"{token_name}: {score}")

    def return_highscore_symbol_and_program(self, env: SRenv.SRenv):
        highscore_symbol = None
        symbol_score = 0

        for token_name, score in self.tokens_score_dict.items():
            # if token_name in env.library.all_token_info_table.token_name_table:
            if token_name not in env.token_args["variable_tokens"] and token_name not in env.token_args["combination_tokens"]:
                continue
            if score > symbol_score:
                symbol_score = score
                highscore_symbol = token_name
                highscore_program = None
                program_score = 0
                for prefix_sequence in self.db.keys():
                    if highscore_symbol in prefix_sequence and self.db[prefix_sequence].reward_dict[
                        "sum_reward"] > program_score:
                        highscore_program = prefix_sequence
                        program_score = self.db[prefix_sequence].reward_dict["sum_reward"]

        new_token_args = env.new_token_args
        symbol_info = new_token_args[highscore_symbol]
        program_info = self.db[highscore_program]

        return symbol_info, program_info

    def return_topk_symbols(self, k):
        sorted_score = sorted(self.tokens_score_dict.items(), key=lambda x: x,
                              reverse=True)
        topk_symbols = []
        for i in range(k):
            topk_symbols.append(sorted_score[i][0])
        return topk_symbols

    def return_topk_expressions(self, topk_num):
        sorted_score = sorted(self.db.items(), key=lambda x: x[1].reward_dict["sum_reward"],
                              reverse=True)
        '''need prefix_expressions'''
        topk_expressions = []
        idx = 0
        if len(sorted_score) == 0:
            return []
        while True:
            prefix_sequence = sorted_score[idx][0]
            if self.db[prefix_sequence].is_reflected:
                idx += 1
                continue
            topk_expressions.append(sorted_score[idx][0])
            self.db[prefix_sequence].is_reflected = True
            idx += 1
            if len(topk_expressions) == topk_num:
                break
            if idx == len(sorted_score):
                break

        topk_expressions_prefix = []
        for expression in topk_expressions:
            prefix = [self.token_names_lib[token_idx] for token_idx in self.db[expression].token_idx]
            topk_expressions_prefix.append(prefix)
        return topk_expressions_prefix


if __name__ == "__main__":
    print("hello".encode())
