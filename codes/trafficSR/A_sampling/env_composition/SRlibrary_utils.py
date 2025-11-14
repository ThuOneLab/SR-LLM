import copy
import numpy as np
import torch
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken import UNITS_VECTOR_SIZE

# Case-code for when units analysis was not performed.
UNITS_ANALYSIS_NOT_PERFORMED_CASE_CODE = 0

# Case-code for when units analysis was not performed.
ILLEGAL_RELATIVE_INDEX = 99999

MAX_CHILDREN_NUMBER = 2


class FreeConstantsTable:
    """
    Contains free constants values.
    """

    def __init__(self, batch_size, n_data_sources, free_constants_init_val):
        # Initial values
        self.init_val = free_constants_init_val.detach().cpu().numpy()

        # Shape
        self.batch_size = batch_size
        self.n_data_sources = n_data_sources
        self.n_free_const = len(free_constants_init_val)  # Number of free constants

        # Free constants values for each program # as torch tensor for fast computation (sent to device in Batch)
        self.values_array = np.tile(self.init_val,
                                    reps=(self.batch_size, self.n_data_sources,
                                          1))  # (batch_size, n_data_sources, n_free_const,) of float
        self.init_val_tensor = torch.tensor(np.tile(self.init_val,
                                                    reps=(self.n_data_sources,
                                                          1)))  # (n_data_sources, n_free_const,) of float
        # If free_constants_init_val already contains torch tensors, they are converted by np.tile (if on same device)
        self.values = torch.tensor(copy.deepcopy(self.values_array))

    def _reset(self):
        self.values = torch.tensor(copy.deepcopy(self.values_array))


class AllTokenInfoTable:
    def __init__(self):
        # Info table
        self.token_name_table, self.is_constraining_phy_units_table, self.phy_units_table, self.arity_table, self.complexity_table, self.type_table, self.power_table, self.dimension_analysis_case_table = np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.length_table = np.array([], dtype=int)

        # arity and type idx group
        self.arity_0_group, self.arity_1_group, self.arity_2_group, self.arity_end_group, self.arity_0_no_combination_group = [], [], [], [], []  # for arity-related prior
        self.operator_group, self.variable_group, self.free_const_group, self.semi_free_const_group, \
            self.fixed_const_group, self.end_group, self.combination_group = [], [], [], [], [], [], []  # for type-related prior

        self.inverse_token_id_table = np.array([])
        self.transcendental_group = []
        self.invalid_id = len(self.token_name_table)  # id of invalid character, end_id+1

        self.const_lb = []
        self.const_ub = []


class SuperParentInfo:
    def __init__(self, max_time_step, library_args):
        self.max_time_step = max_time_step
        self.superparent_units = np.full(UNITS_VECTOR_SIZE, 0.)
        self.superparent_units[:len(np.array(library_args["superparent_units"]))] = np.array(
            library_args["superparent_units"])
        self.superparent_is_constraining_phy_units = True
        self.superparent_names = [] if library_args["superparent_names"] is None else library_args["superparent_names"]
        self.superparent_prog = [] if library_args["superparent_prog"] is None else library_args["superparent_prog"]
        self.superparent_number = len(self.superparent_prog)
        self.anchor_prog = []
        self.total_complexity = 0

        #  ---- relative info ----
        self.parent_index = np.full(shape=(len(self.superparent_prog), self.max_time_step),
                                    fill_value=ILLEGAL_RELATIVE_INDEX,
                                    dtype=int)
        self.sibling_index = np.full(shape=(len(self.superparent_prog), self.max_time_step),
                                     fill_value=ILLEGAL_RELATIVE_INDEX,
                                     dtype=int)
        self.children_index = np.full(shape=(len(self.superparent_prog), self.max_time_step, MAX_CHILDREN_NUMBER),
                                      fill_value=ILLEGAL_RELATIVE_INDEX,
                                      dtype=int)
        self.children_number = np.full(shape=(len(self.superparent_prog), self.max_time_step),
                                       fill_value=0,
                                       dtype=int)
        self.children_end = np.full(shape=(len(self.superparent_prog), self.max_time_step, MAX_CHILDREN_NUMBER),
                                    fill_value=ILLEGAL_RELATIVE_INDEX,
                                    dtype=int)
        self.children_length = np.full(shape=(len(self.superparent_prog), self.max_time_step, MAX_CHILDREN_NUMBER),
                                       fill_value=ILLEGAL_RELATIVE_INDEX,
                                       dtype=int)
        # self.depth = np.full(shape=(1, self.max_time_step), fill_value=ILLEGAL_RELATIVE_INDEX, dtype=int)
        self.height = np.full(shape=(len(self.superparent_prog), self.max_time_step), fill_value=0, dtype=int)


class LLMInfo:
    def __init__(self, shape):
        self.shape = shape
        self.use_llm = np.full(shape, fill_value=False, dtype=bool)
        self.llm_tokens_name = np.full(shape, fill_value="", dtype='<U32')
        self.tokens_id = np.full(shape, fill_value=ILLEGAL_RELATIVE_INDEX, dtype=int)

    def _reset(self):
        self.use_llm = np.full(self.shape, fill_value=False, dtype=bool)
        self.llm_tokens_name = np.full(self.shape, fill_value="", dtype='<U32')
        self.tokens_id = np.full(self.shape, fill_value=ILLEGAL_RELATIVE_INDEX, dtype=int)

    # def get_llm_template(self, llm: templates):
    #     self.llm = llm


class CombinationInfo:
    def __init__(self, shape):
        self.shape = shape
        self.have_combination = np.full(shape=self.shape[0], fill_value=False, dtype=bool)
        self.combination_tokens_id = np.full(self.shape, fill_value=ILLEGAL_RELATIVE_INDEX, dtype=int)
        self.height_table, self.parent_index_table, self.sibling_index_table, self.children_index_table, self.children_number_table, self.children_end_table = [], [], [], [], [], []

    def _reset(self):
        self.have_combination = np.full(shape=self.shape[0], fill_value=False, dtype=bool)
        self.combination_tokens_id = np.full(self.shape, fill_value=ILLEGAL_RELATIVE_INDEX, dtype=int)
        
class PostPhysicalCheck:
    def __init__(self,min_occupancy_times):
        self.targets_name=min_occupancy_times["targets_name"]
        self.min_times=min_occupancy_times["min_times"]
        
    def get_target_id(self,all_tokens_id_dict):
        if len(self.targets_name)>0:
            self.targets_id=[all_tokens_id_dict[target] for target in self.targets_name]
    
    # 采样结束后，进行occupancy的物理检查
    def post_physical_check_of_occupancy(self,is_physical,have_completed,tokens_idx):
        if len(self.targets_name)==0:
            return False, is_physical
        mask_need_check = copy.deepcopy(is_physical & have_completed)  # mask: should calculate reward，不满足物理约束的就不用计算reward
        prog_ids = np.where(mask_need_check)[0]
        tokens_idx_need_check = tokens_idx[prog_ids]
        for i, target_id in enumerate(self.targets_id):
            mask_target = tokens_idx_need_check == target_id
            mask_target = mask_target.sum(axis=1) >= self.min_times[i] # 至少出现min_times[i]次
            is_physical[prog_ids[~mask_target]] = False # 说明不满足至少出现min_times[i]次的约束，之后就不计算reward了，直接reward=0
        return True, is_physical