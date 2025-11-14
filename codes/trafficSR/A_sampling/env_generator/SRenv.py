import copy
import numpy as np
import torch
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken import Token
from codes.trafficSR.A_sampling.env_composition.SRprograms import SRprograms
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken_combinations import Token_combination
from codes.trafficSR.A_sampling.env_composition.SRlibrary import SRlibrary, ILLEGAL_RELATIVE_INDEX
from codes.trafficSR.A_sampling.policy_network.policy_network import PolicyNetwork
from codes.trafficSR.A_sampling.prior.SRprior import DEFAULT_INVERSE_OPERATOR, DEFAULT_TRANSCENDENTAL_OPERATOR

def get_list_from_dict(dict,key_list):
    return [dict[key] for key in key_list]

class SRenv:
    def __init__(
            self,
            X_numpy,
            y_numpy,
            library_args,
            env_args,
            token_args,
            ngsim_args=None,
            bool_args=None,
            prefix_tree=None,
            seed=100,
            batch_size=None,
    ):
        self.X_numpy = X_numpy
        self.y_numpy = y_numpy
        self.ngsim_args = ngsim_args
        self.env_args = env_args
        self.token_args = token_args
        # self.new_token_args = self.rebase_token_args(token_args)

        self.batch_size = env_args['batch_size'] if batch_size is None else batch_size
        self.max_time_step = env_args['max_time_step']

        self.risk_factor = env_args['risk_factor']
        self.device = env_args['device']
        self.dtype = env_args['dtype']
        self.seed = seed

        # bool settings
        self.bool_use_combinations = bool_args['bool_use_combinations']
        self.bool_use_rough_calibration = bool_args['bool_use_rough_calibration']
        self.bool_use_UCB_for_sampling = bool_args['bool_use_UCB_for_sampling']

        self.library_args = library_args
        self.library = SRlibrary((self.batch_size, self.max_time_step), library_args)
        self.prefix_tree = prefix_tree
        
        token_args['free_const_initial_values']=get_list_from_dict(token_args['free_const_initial_values_dict'],token_args["free_const_tokens"])
        self.free_const_values = torch.tensor(token_args['free_const_initial_values']).to(device=self.device)
        self.free_const_values.requires_grad = True
        
        token_args['semi_free_const_initial_values']=get_list_from_dict(token_args['semi_free_const_initial_values_dict'],token_args["semi_free_const_tokens"])
        self.semi_free_const_values = torch.tensor(token_args['semi_free_const_initial_values']).to(device=self.device)
        self.semi_free_const_values.requires_grad = True
        
        token_args['fixed_const_values']=get_list_from_dict(token_args['fixed_const_values_dict'],token_args["fixed_const_tokens"])
        self.fixed_const_values = torch.tensor(token_args['fixed_const_values']).to(device=self.device)
        self.fixed_const_values.requires_grad = False

        self._init_all_tokens()
        self.rng = np.random.RandomState(seed)
        self.programs = SRprograms(env_args=self.env_args, library=self.library, all_tokens=self.all_tokens,
                                   free_const_values=self.free_const_values,
                                   semi_free_const_values=self.semi_free_const_values,
                                   fixed_const_values=self.fixed_const_values, zero_out_unphysical=True,
                                   n_data_sources=self.ngsim_args[
                                       "n_data_sources"] if self.ngsim_args is not None else 1,
                                   bool_args=bool_args,
                                   batch_size=self.batch_size,
                                   rng=self.rng)
        # self.programs.invalid_id = self.library.all_tokens_id_dict["end"]
        self._reset()

    def getTraj(self, train_args, policy_network:PolicyNetwork, prior_collector, prefix_tree, epoch, specific_program=None):
        '''init states'''
        logits = []
        actions = []
        observations_overall, observations_partial = [], []
        states_overall, states_partial = [], []
        prior_UCBs = []
        states = None if train_args['model_type'] == 'transformer' \
            else policy_network.model.get_zeros_initial_state(self.batch_size).to(self.device, self.dtype)
        state_overall = policy_network.model.get_zeros_initial_state(self.batch_size, overall_or_partial=0).to(
            self.device, self.dtype)
        state_partial = policy_network.model.get_zeros_initial_state(self.batch_size, overall_or_partial=1).to(
            self.device, self.dtype)
        
        if specific_program is not None:
            specific_prog_prob = 1

        '''time step iteration'''
        for time_id in range(self.max_time_step):
            '''get model_prob'''
            observations = policy_network.get_observations(env=self)
            observations_overall.append(observations[0])
            observations_partial.append(observations[1])
            states_overall.append(state_overall)
            states_partial.append(state_partial)
            output, states, state_partial, state_overall = policy_network.output_and_state(states, state_partial,
                                                                                           state_overall,
                                                                                           time_id, self.batch_size,
                                                                                           observations)
            while torch.any(torch.isnan(output)):#有一次出现全为nan的情况，导致后面的都是nan而报错
                print("policy_network output contains nan")
                observations = policy_network.get_observations(env=self)
                observations_overall.append(observations[0])
                observations_partial.append(observations[1])
                states_overall.append(state_overall)
                states_partial.append(state_partial)
                output, states, state_partial, state_overall = policy_network.output_and_state(states, state_partial, state_overall,
                                                                                            time_id, self.batch_size,
                                                                                            observations)
            model_prob = output.to(self.device, self.dtype)  # (batch_size, output_size)
            '''get prior_prob'''
            prior_prob = torch.tensor(self.get_prior(prior_collector),requires_grad=False).to(self.device,
                                                                          self.dtype)  # (batch_size, output_size)
            logprior = torch.log(prior_prob)  # [batchsize,action_dim]——可能不加这一步会更好。不如做一步softmax
            '''get N_of_edge_batch'''
            N_of_edge_batch = torch.tensor(
                prefix_tree.N_of_edge_batch_new(self.library)).to(self.device,
                                                                  self.dtype)  # every token show times in a expression
            '''get action prob'''
            UCB = torch.sqrt(0.25 * np.log(epoch + 1) / (N_of_edge_batch + 1))  # [batchsize,action_dim] 2用于跟驰，增强探索；0.25用于普通情况
            action_prob = (model_prob + logprior) + UCB if self.bool_use_UCB_for_sampling else model_prob + logprior # model_prob主要输出模型的概率;logprior主要用来惩罚，是-inf到0的数;UCB主要用来探索
            # action_prob = (model_prob + logprior) / (1 + 100 * N_of_edge_batch ** 2)
            '''get exp_prob, 之后看看mask_tuple_tensor是否有问题（如果全部符号都不能选择，就只能随机取了）'''
            exp_prob = torch.exp(action_prob)
            mask_tuple_tensor = torch.where(torch.all(exp_prob == 0, dim=1)) #发现bug本质是因为：它随机取了一个符号，但是这个符号本来是不能加的（因为加了的话，会导致arity+1或者+2），导致超出长度限制LengthPrior。比如第277个。
            inf_mask = mask_tuple_tensor[0] if type(mask_tuple_tensor) == tuple else mask_tuple_tensor
            if len(inf_mask) > 0:
                exp_prob[inf_mask,:-1] = 1  # if prob all 0, then action should be randomly selected.但是不能随机取end，会导致提前结束
                exp_prob[inf_mask, -1] = 0
            '''get action from exp_prob'''
            # torch.multinomial: sample indices from a multinomial distribution based on the input tensor probabilities

            action = torch.multinomial(exp_prob, num_samples=1)[:, 0]
            if specific_program is not None:
                if time_id > len(specific_program) - 1:
                    action = torch.tensor([self.library.all_tokens_id_dict["end"]] * self.batch_size).to(self.device)
                else:
                    action = torch.tensor([self.library.all_tokens_id_dict[specific_program[time_id]]] * self.batch_size).to(
                        self.device)
                    assert exp_prob[0, self.library.all_tokens_id_dict[specific_program[time_id]]] != 0, "specific_program is illegal"
                    this_prob_normalized = exp_prob[0, self.library.all_tokens_id_dict[specific_program[time_id]]] / torch.sum(exp_prob[0])
                    specific_prog_prob *= this_prob_normalized

                if time_id == self.max_time_step - 1:
                    print("specific_program is legal!", f"prog_prob:{specific_prog_prob}")
                if time_id < len(specific_program):
                    print("specific_program is legal at time_id:", time_id, ", token:", specific_program[time_id], ", token_prob:", this_prob_normalized,  ", current program:", specific_program[:time_id + 1])
            
            actions.append(action)
            logits.append(action_prob)  # model+prior is best
            prior_UCB = logprior + UCB if self.bool_use_UCB_for_sampling else logprior
            prior_UCBs.append(prior_UCB)
            '''env step'''
            self.step(action=action, )

        '''get logits, actions'''
        logits = torch.stack(logits, dim=0)  # (max_time_step, batch_size, tokens_number, )
        actions = torch.stack(actions, dim=0)  # (max_time_step, batch_size,)
        actions_numpy = actions.detach().cpu().numpy()  # (max_time_step, batch_size,)
        observations_overall = torch.stack(observations_overall, dim=0)
        observations_partial = torch.stack(observations_partial, dim=0)
        states_overall = torch.stack(states_overall, dim=0)
        states_partial = torch.stack(states_partial, dim=0)
        prior_UCBs = torch.stack(prior_UCBs, dim=0)
        
        '''post physical and rewards'''
        n_is_physical_origin=self.library.is_physical.sum()
        need_check,self.library.is_physical=self.library.post_physical_check.post_physical_check_of_occupancy(self.library.is_physical,self.library.have_completed,self.library.tokens_idx)
        if need_check:
            print("is_physical origin and after",n_is_physical_origin,self.library.is_physical.sum())
        R, R_SUB, R_similarity = self.get_reward(prefix_tree=prefix_tree)

        if specific_program is not None:
            print(f"\nreward of specific_program:{R.max()}\n")

        return (logits, actions_numpy, observations_overall, observations_partial, R, R_SUB,
                R_similarity, observations, states_overall, states_partial, prior_UCBs)

    def _init_all_tokens(
            self,
    ):
        '''
        variable, fixed_const, free_const, operator, end
        '''

        def append_library_table(token):
            self.library.all_token_info_table.token_name_table = np.append(
                self.library.all_token_info_table.token_name_table, token.token_name)
            self.library.all_token_info_table.phy_units_table = np.append(
                self.library.all_token_info_table.phy_units_table, token.phy_units,
                axis=0)  # by row; no axis variable will lead to append at last
            self.library.all_token_info_table.is_constraining_phy_units_table = np.append(
                self.library.all_token_info_table.is_constraining_phy_units_table,
                token.is_constraining_phy_units)
            self.library.all_token_info_table.arity_table = np.append(self.library.all_token_info_table.arity_table,
                                                                      token.token_arity)
            self.library.all_token_info_table.complexity_table = np.append(
                self.library.all_token_info_table.complexity_table,
                token.complexity)
            if token.token_arity >= 0:
                eval(f"self.library.all_token_info_table.arity_{token.token_arity}_group").append(token.token_id)
            else:
                self.library.all_token_info_table.arity_end_group.append(token.token_id)
            self.library.all_token_info_table.type_table = np.append(self.library.all_token_info_table.type_table,
                                                                     token.token_type_kind_number)
            self.library.all_token_info_table.power_table = np.append(self.library.all_token_info_table.power_table,
                                                                      token.power)
            self.library.all_token_info_table.dimension_analysis_case_table = np.append(
                self.library.all_token_info_table.dimension_analysis_case_table,
                token.dimension_analysis_case)
            eval(f"self.library.all_token_info_table.{token.token_type}_group").append(token.token_id)
            self.library.all_token_info_table.length_table = np.append(self.library.all_token_info_table.length_table,
                                                                       token.length)
            if token.token_type == "combination":
                self.library.combination_info.height_table.append(list(token.height))
                self.library.combination_info.parent_index_table.append(list(token.parent_index))
                self.library.combination_info.sibling_index_table.append(list(token.sibling_index))
                self.library.combination_info.children_index_table.append(list(token.children_index))
                self.library.combination_info.children_number_table.append(list(token.children_number))
                self.library.combination_info.children_end_table.append(list(token.children_end))
            else:
                self.library.combination_info.height_table.append([1])
                self.library.combination_info.parent_index_table.append([])
                self.library.combination_info.sibling_index_table.append([])
                self.library.combination_info.children_index_table.append([])
                self.library.combination_info.children_number_table.append([])
                self.library.combination_info.children_end_table.append([])

        def update_library_inverse_token_id_table():
            self.library.all_token_info_table.inverse_token_id_table = np.full(
                shape=(self.tokens_number), fill_value=ILLEGAL_RELATIVE_INDEX,
                dtype=int)
            for key in self.library.all_tokens_id_dict:
                if key in DEFAULT_INVERSE_OPERATOR and DEFAULT_INVERSE_OPERATOR[
                    key] in self.library.all_token_info_table.token_name_table:
                    self.library.all_token_info_table.inverse_token_id_table[self.library.all_tokens_id_dict[key]] = \
                        self.library.all_tokens_id_dict[DEFAULT_INVERSE_OPERATOR[key]]

        def update_library_transcendental_token_id_table():
            for key in self.library.all_tokens_id_dict:
                if key in DEFAULT_TRANSCENDENTAL_OPERATOR:
                    self.library.all_token_info_table.transcendental_group.append(self.library.all_tokens_id_dict[key])

        self.all_tokens = []
        self.library.all_tokens_id_dict = {}
        this_id = 0
        
        # init operators
        operator_tokens_names = self.token_args['operator_tokens']
        for on in operator_tokens_names:
            tok = Token(
                name=on,
                type='operator',
                id=this_id,
            )
            self.all_tokens.append(
                tok
            )
            self.library.all_tokens_id_dict[on] = tok.token_id
            append_library_table(tok)
            this_id += 1
        
        # init fixed constants
        fixed_tokens_names = self.token_args['fixed_const_tokens']
        fixed_tokens_units = self.token_args["fixed_const_units"]
        fixed_tokens_description = self.token_args["fixed_const_description"]
        fixed_id=0
        for fn in fixed_tokens_names:
            tok = Token(
                name=fn,
                type='fixed_const',
                id=this_id,
                fixed_value=self.token_args['fixed_const_values'][fixed_id],
                phy_units=fixed_tokens_units[fn],
                description=fixed_tokens_description[fn]
            )
            self.all_tokens.append(tok)
            self.library.all_tokens_id_dict[fn] = tok.token_id
            append_library_table(tok)
            this_id += 1
            fixed_id+=1
            
        # init semi_free constants
        semi_free_tokens_names = self.token_args['semi_free_const_tokens']
        semi_free_tokens_units = self.token_args["semi_free_const_units"]
        semi_free_tokens_bounds = self.token_args["semi_free_const_bounds"]
        semi_free_tokens_description = self.token_args["semi_free_const_description"]
        for fn in semi_free_tokens_names:
            tok = Token(
                name=fn,
                type='semi_free_const',
                id=this_id,
                phy_units=semi_free_tokens_units[fn],
                description=semi_free_tokens_description[fn]
            )
            self.all_tokens.append(tok)
            self.library.all_tokens_id_dict[fn] = tok.token_id
            append_library_table(tok)
            self.library.all_token_info_table.const_lb.append(semi_free_tokens_bounds[fn][0])
            self.library.all_token_info_table.const_ub.append(semi_free_tokens_bounds[fn][1])
            this_id += 1
            
        # init free constants
        free_tokens_names = self.token_args['free_const_tokens']
        free_tokens_units = self.token_args["free_const_units"]
        free_tokens_bounds = self.token_args["free_const_bounds"]
        free_tokens_description = self.token_args["free_const_description"]
        for fn in free_tokens_names:
            tok = Token(
                name=fn,
                type='free_const',
                id=this_id,
                phy_units=free_tokens_units[fn],
                description=free_tokens_description[fn]
            )
            self.all_tokens.append(tok)
            self.library.all_tokens_id_dict[fn] = tok.token_id
            append_library_table(tok)
            self.library.all_token_info_table.const_lb.append(free_tokens_bounds[fn][0])
            self.library.all_token_info_table.const_ub.append(free_tokens_bounds[fn][1])
            this_id += 1

        # init input variables
        variable_tokens_names = self.token_args['variable_tokens']
        variable_tokens_units = self.token_args["variable_units"]
        variable_tokens_description = self.token_args["variable_description"]
        for vn in variable_tokens_names:
            tok = Token(
                name=vn,
                type='variable',
                id=this_id,
                phy_units=variable_tokens_units[vn],
                description=variable_tokens_description[vn]
            )
            self.all_tokens.append(tok)
            self.library.all_tokens_id_dict[vn] = tok.token_id
            append_library_table(tok)
            this_id += 1

        if self.bool_use_combinations:
            # init operators
            combination_tokens_names = copy.deepcopy(self.token_args['combination_tokens'])
            combination_prefix_expression = self.token_args['combination_prefix_expression']
            combination_tokens_units = self.token_args["combination_units"]
            combination_descriptions = self.token_args["combination_description"]
            for cn in combination_tokens_names:
                tok = Token_combination(
                    all_tokens=self.all_tokens,
                    library=self.library,
                    name=cn,
                    type='combination',
                    id=this_id,
                    prefix_expression=combination_prefix_expression[cn],
                    phy_units=combination_tokens_units[cn],
                    description=combination_descriptions[cn],
                )
                if tok.legal == False:  # combination not legal
                    self.token_args['combination_tokens'].remove(cn)
                    self.token_args["combination_units"].pop(cn)
                    self.token_args["combination_description"].pop(cn)
                    self.token_args['combination_prefix_expression'].pop(cn)
                else:
                    self.all_tokens.append(tok)
                    self.library.all_tokens_id_dict[cn] = tok.token_id
                    self.token_args["combination_units"][cn] = list(tok.phy_units)
                    append_library_table(tok)
                    this_id += 1

        # placeholder_tok = Token(
        #     name='placeholder',
        #     type='end',
        #     id=this_id,
        # )
        # self.all_tokens.append(
        #     placeholder_tok
        # )
        # self.library.all_tokens_id_dict['placeholder'] = placeholder_tok.token_id
        # append_library_table(placeholder_tok)
        # this_id += 1

        # init end token
        end_tok = Token(
            name='end',
            type='end',
            id=this_id,
        )
        self.all_tokens.append(
            end_tok
        )
        self.library.all_tokens_id_dict['end'] = end_tok.token_id
        append_library_table(end_tok)
        this_id += 1

        self.tokens_number = len(self.all_tokens)
        self.library.tokens_number = len(self.all_tokens)
        self.library.all_token_info_table.phy_units_table = np.reshape(
            self.library.all_token_info_table.phy_units_table, (-1, 7))
        self.library.all_token_info_table.is_constraining_phy_units_table = self.library.all_token_info_table.is_constraining_phy_units_table.astype(
            bool)
        self.library.all_token_info_table.invalid_id = len(self.all_tokens)
        self.library.all_token_info_table.arity_0_no_combination_group = self.library.all_token_info_table.variable_group + self.library.all_token_info_table.free_const_group + \
                                                                         self.library.all_token_info_table.semi_free_const_group + self.library.all_token_info_table.fixed_const_group
        update_library_inverse_token_id_table()
        update_library_transcendental_token_id_table()
        self.library.post_physical_check.get_target_id(all_tokens_id_dict=self.library.all_tokens_id_dict)

    def _reset(
            self,
    ):
        self.library._reset()
        self.programs._reset()
        self.programs.bool_use_rough_calibration = self.bool_use_rough_calibration

    def update_N_table(self, new_tokens_idx, ):
        self.prefix_tree.update_N_table_every_timestep(library=self.library, new_tokens_idx=new_tokens_idx)

    def step(
            self,
            action,
    ):
        assert action.shape[0] == self.batch_size, f"action.shape[0] must be equal to batch_size"
        action_cpu = action.cpu() if action.device != 'cpu' else action
        self.programs.append(action_cpu)
        self.update_N_table(action_cpu)
        self.library.update_prefix_str(action_cpu)

    def get_observation(
            self,
    ):
        return self.programs.library.tokens_idx  # (batch_size, max_time_step)

    def get_overall_observation(
            self,
    ):
        # -------------------placeholder(not good)
        # result = copy.deepcopy(self.programs.library.tokens_idx)
        # mask_have_placeholder = self.programs.library.n_placeholder_now > 0
        # min_complete_length = self.programs.library.n_placeholder_now + self.programs.valid_lengths
        #
        # batch_pos = np.tile(np.arange(0, self.max_time_step), (self.batch_size, 1)).astype(int)
        # mask_placeholder = np.logical_and(
        #     batch_pos >= np.tile(self.programs.valid_lengths, (self.max_time_step, 1)).transpose(),
        #     batch_pos < np.tile(min_complete_length, (self.max_time_step, 1)).transpose())
        # mask_placeholder = np.logical_and(mask_placeholder, np.tile(mask_have_placeholder, (self.max_time_step,1)).transpose())
        # result[mask_placeholder] = self.programs.invalid_id + 1

        return self.programs.library.tokens_idx  # (batch_size, max_time_step)
        # return self.programs.library.tokens_one_hot

    def get_partial_observation(
            self,
            mode="one_hot",
    ):
        # -------------------relation and unit info
        # Do run only on incomplete programs AND physical(?)
        mask_do_run = (~self.library.have_completed) & (self.library.is_physical)
        # mask_do_run = (~self.library.have_completed)
        batch_id = np.arange(start=0, stop=self.batch_size, step=1)
        coords = np.stack(
            (batch_id[mask_do_run], np.full(shape=sum(mask_do_run), fill_value=self.library.current_time_step)), axis=0)
        # print("n_progs_do_get_obs:", sum(mask_do_run))

        current_units_obs = self.programs.initialize_unit_obs(self.batch_size)
        current_units_obs[mask_do_run] = self.programs.get_current_info(
            step=self.library.current_time_step, coords=coords)

        parent_idx_label = np.full(shape=(self.batch_size), fill_value=self.library.all_token_info_table.invalid_id,
                                   dtype=np.float32)
        parent_one_hot = np.zeros((self.batch_size, self.tokens_number))
        parent_units_obs = self.programs.initialize_unit_obs(self.batch_size)
        parent_idx_label[mask_do_run], parent_one_hot[mask_do_run], parent_units_obs[
            mask_do_run] = self.programs.get_parent_info(
            step=self.library.current_time_step, coords=coords)

        sibling_idx_label = np.full(shape=(self.batch_size), fill_value=self.library.all_token_info_table.invalid_id,
                                    dtype=np.float32)
        sibling_one_hot = np.zeros((self.batch_size, self.tokens_number))
        sibling_units_obs = self.programs.initialize_unit_obs(self.batch_size)
        sibling_idx_label[mask_do_run], sibling_one_hot[mask_do_run], sibling_units_obs[
            mask_do_run] = self.programs.get_sibling_info(
            step=self.library.current_time_step, coords=coords)

        previous_idx_label = np.full(shape=(self.batch_size), fill_value=self.library.all_token_info_table.invalid_id,
                                     dtype=np.float32)
        previous_one_hot = np.zeros((self.batch_size, self.tokens_number))
        previous_units_obs = self.programs.initialize_unit_obs(self.batch_size)
        if self.library.current_time_step > 0:
            previous_idx_label[mask_do_run], previous_one_hot[mask_do_run], previous_units_obs[
                mask_do_run] = self.programs.get_previous_info(
                step=self.library.current_time_step, coords=coords)

        n_placeholder_now = self.library.n_placeholder_now

        if mode == "one_hot":
            result = np.concatenate((  # (batch_size, obs_size,)
                # tokens
                # tokens_idx,
                # parent_idx_label[:, np.newaxis],
                # sibling_idx_label[:, np.newaxis],
                # previous_idx_label[:, np.newaxis],
                # Relatives one-hots
                parent_one_hot,
                sibling_one_hot,
                previous_one_hot,
                # Number of dangling dummies
                n_placeholder_now[:, np.newaxis],  # (batch_size,)
                # Units obs
                current_units_obs,
                parent_units_obs,
                sibling_units_obs,
                previous_units_obs,
            ), axis=1).astype(np.float32)
        else:
            result = np.concatenate((  # (batch_size, obs_size,)
                # tokens
                # tokens_idx,
                parent_idx_label[:, np.newaxis],
                sibling_idx_label[:, np.newaxis],
                previous_idx_label[:, np.newaxis],
                # Relatives one-hots
                # parent_one_hot,
                # sibling_one_hot,
                # previous_one_hot,
                # Number of dangling dummies
                n_placeholder_now[:, np.newaxis],  # (batch_size,)
                # Units obs
                current_units_obs,
                parent_units_obs,
                sibling_units_obs,
                previous_units_obs,
            ), axis=1).astype(np.float32)

        return result

    def get_prior(
            self, prior_collector
    ):
        if len(prior_collector.prior_types) == 0:
            prior = np.ones((self.programs.batch_size, self.tokens_number))
        else:
            prior = prior_collector()
        return prior

    def get_reward(
            self,
            prefix_tree,
    ):
        result = self.programs.get_reward_new(self.X_numpy, self.y_numpy, prefix_tree=prefix_tree,
                                              ngsim_args=self.ngsim_args,
                                              parrallel_mode=self.env_args['parallel_mode'],
                                              n_cpus=self.env_args['n_cpus'],
                                              device=self.device,
                                              )
        return result

    def get_minimal_expression_of_superparents(self):
        superparents = self.library_args["superparent_prog"]
        prefix_expression_dict = {}
        token_name_list = list(self.library.all_tokens_id_dict.keys())
        for tok in self.all_tokens:
            prefix_expression_dict[tok.token_name] = tok.prefix_expression

        minimal_expressions = []
        for superparent in superparents:
            matching_results_str = []
            matching_results_idx = []
            matching_idx = 0
            while True:
                sum_arity = self.all_tokens[self.library.all_tokens_id_dict[superparent[matching_idx]]].token_arity
                this_idx = matching_idx
                while True:
                    if sum_arity == 0:
                        matching_token = superparent[matching_idx: this_idx + 1]
                        if len(matching_token) == 1:
                            matching_results_str.append(superparent[matching_idx])
                            matching_results_idx.append(self.library.all_tokens_id_dict[superparent[matching_idx]])
                            matching_idx += 1
                        elif matching_token in prefix_expression_dict.values():
                            for name, prefix_expression in prefix_expression_dict.items():
                                if matching_token == prefix_expression:
                                    matching_results_str.append(name)
                                    matching_results_idx.append(self.library.all_tokens_id_dict[name])
                                    break
                            matching_idx += len(matching_token)
                        else:
                            matching_results_str.append(superparent[matching_idx])
                            matching_results_idx.append(self.library.all_tokens_id_dict[superparent[matching_idx]])
                            matching_idx += 1
                        break
                    this_idx += 1
                    if this_idx == len(superparent):
                        raise ValueError("superparent is illegal!")
                    sum_arity += self.all_tokens[self.library.all_tokens_id_dict[superparent[this_idx]]].token_arity - 1
                if matching_idx == len(superparent):
                    break

            while len(matching_results_str) < self.max_time_step:
                matching_results_str.append("end")
                matching_results_idx.append(self.library.all_tokens_id_dict["end"])

            minimal_expressions.append({
                "superparent": superparent,
                "matching_results_str": matching_results_str,
                "matching_results_idx": matching_results_idx
            })

        return minimal_expressions

    def rebase_token_args(self, token_args):
        token_types = ["variable", "operator", "free_const", "fixed_const", "combination"]
        new_token_args = {}
        for token_type in token_types:

            for token_id in range(len(token_args[f"{token_type}_tokens"])):
                token_name = token_args[f"{token_type}_tokens"][token_id]
                token_description = token_args[f"{token_type}_description"][token_name]
                token_unit = token_args[f"{token_type}_units"][token_name][:2] if token_type != "operator" else None
                token_prefix_expression = token_args[f"{token_type}_prefix_expression"][
                    token_name] if token_type == "combination" else [token_name]
                new_token_args[token_name] = {
                    "name": token_name,
                    "description": token_description,
                    "units": token_unit,
                    "prefix_expression": token_prefix_expression,
                    "type": token_type,
                }

        return new_token_args
