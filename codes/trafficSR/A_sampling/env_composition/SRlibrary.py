import numpy as np
from codes.trafficSR.A_sampling.env_tokens_combinations.SRtoken import UNITS_VECTOR_SIZE, \
    TYPE_KIND_NUMBER  # Number of units in SI system
from codes.trafficSR.A_sampling.env_composition.SRlibrary_utils import UNITS_ANALYSIS_NOT_PERFORMED_CASE_CODE, \
    ILLEGAL_RELATIVE_INDEX, \
    MAX_CHILDREN_NUMBER, AllTokenInfoTable, SuperParentInfo, LLMInfo, CombinationInfo, PostPhysicalCheck


class SRlibrary:
    def __init__(
            self,
            shape,
            library_args
    ):
        self.current_time_step = 0
        self.shape = shape  # (int, int) (batch_size,max_time_step)
        self.batch_size = self.shape[0]
        self.max_time_step = self.shape[1]

        # Properties : position is the same in all elements of batch
        self.pos = np.tile(np.arange(0, self.shape[1]), (self.shape[0], 1)).astype(int)
        self.pos_batch = np.tile(np.arange(0, self.shape[0]), (self.shape[1], 1)).transpose().astype(int)

        # info table
        self.all_token_info_table = AllTokenInfoTable()
        self.super_parent_info = SuperParentInfo(self.max_time_step, library_args)
        self.llm_info = LLMInfo(self.shape)
        self.combination_info = CombinationInfo(self.shape)
        self.post_physical_check = PostPhysicalCheck(library_args["min_occupancy_times"])
        self.tokens_number = 0

    # reset when every epoch begins
    def _reset(self):
        self.combination_info._reset()
        #  ---- tokens  ----
        self.tokens_idx = np.ones(self.shape, dtype=np.int64) * self.all_token_info_table.invalid_id
        self.tokens_one_hot = np.zeros(
            (self.batch_size, self.max_time_step * len(self.all_token_info_table.token_name_table)), dtype=np.int64)
        self.eye = np.eye(self.tokens_number)
        self.real_length = np.zeros(self.shape[0], dtype=int)
        self.prefix_str = np.full(self.shape[0], fill_value="", dtype='<U512')
        self.include_free_const = np.full(self.shape[0], fill_value=False, dtype=bool)
        #  ---- physical units  ----
        self.is_constraining_phy_units = np.full(shape=(self.shape[0], self.shape[1]),
                                                 fill_value=False,
                                                 dtype=bool)
        # self.is_constraining_phy_units[:, 0] = True  # superparent
        self.phy_units = np.full(shape=(self.shape[0], self.shape[1], UNITS_VECTOR_SIZE,),
                                 fill_value=np.nan,
                                 dtype=float)
        # self.phy_units[:, 0] = self.superparent_units
        self.units_inconsistency = np.full(shape=self.shape[0], fill_value=-1, dtype=int)
        self.is_physical = np.full(shape=self.shape[0], fill_value=True,
                                   dtype=bool)  # (batch_size,) of bool, for physical units
        self.current_time_step = 0

        self.units_analysis_cases = np.full(shape=self.shape, fill_value=UNITS_ANALYSIS_NOT_PERFORMED_CASE_CODE,
                                            dtype=int)  # (batch_size, max_time_step,) of bool
        self.total_complexity = np.zeros(self.shape[0], dtype=float)  # (batch_size,) of float

        # ---- Completed  ----
        # Individual lengths of programs <= max_time_step,
        # tokens after n_lengths are dummies or do not have meaning
        self.n_lengths = np.zeros(self.shape[0], dtype=int)  # (batch_size,) of int
        # Sum of arities over time dim
        self.total_arities = np.ones(self.shape[0], dtype=float)  # (batch_size,) of int
        self.have_completed = np.full(shape=self.shape[0], fill_value=False,
                                      dtype=bool)
        self.have_end = np.full(shape=self.shape[0], fill_value=False,
                                      dtype=bool) #是否已经有END被加入进去了

        #  ---- relative info ----
        self.parent_index = np.full(shape=(self.shape[0], self.shape[1]),
                                    fill_value=ILLEGAL_RELATIVE_INDEX,
                                    dtype=int)
        self.sibling_index = np.full(shape=(self.shape[0], self.shape[1]),
                                     fill_value=ILLEGAL_RELATIVE_INDEX,
                                     dtype=int)
        self.children_index = np.full(shape=(self.shape[0], self.shape[1], MAX_CHILDREN_NUMBER),
                                      fill_value=ILLEGAL_RELATIVE_INDEX,
                                      dtype=int)
        self.children_number = np.full(shape=(self.shape[0], self.shape[1]),
                                       fill_value=0,
                                       dtype=int)
        self.children_end = np.full(shape=(self.shape[0], self.shape[1], MAX_CHILDREN_NUMBER),
                                    fill_value=ILLEGAL_RELATIVE_INDEX,
                                    dtype=int)
        self.children_length = np.full(shape=(self.shape[0], self.shape[1], MAX_CHILDREN_NUMBER),
                                       fill_value=0,
                                       dtype=int)
        # self.depth = np.full(shape=self.shape, fill_value=ILLEGAL_RELATIVE_INDEX, dtype=int)
        self.height = np.full(shape=self.shape, fill_value=0, dtype=int)

        #  ---- Placeholder info ----
        self.n_placeholder_left = np.zeros(self.batch_size,
                                           dtype=int)  # (batch_size,) of int——after add one token, left how many placeholders
        # Number of tokens needed to finish the program
        self.n_placeholder_now = np.ones(self.batch_size, dtype=int)  # (batch_size,) of int

        self.llm_info._reset()

    '''utils'''

    def mask_to_coords(self, mask):
        """
        Helper function returning coordinates where mask is True.
        Parameters
        ----------
        mask : numpy.array of shape (batch_size, max_time_step) of bool
            Mask.
        Returns
        -------
        mask_sum, coordinates : int, numpy.array of shape (2, mask_sum) of int
            Number of coordinates where mask is True, coordinates where mask is True.
        """
        # Showing that memory space can accurately be allocated before-hand
        mask_sum = mask.sum()  # int
        coords = np.zeros(shape=(2, mask_sum,), dtype=int)  # (2, mask_sum,) of int
        # Coordinates
        coords[:, :] = np.stack((  # (2, mask_sum,) of int
            self.pos_batch[mask],  # batch dim coord
            self.pos[mask],  # time dim coord
        ), axis=0)
        return mask_sum, coords

    def coords_to_mask(self, coords):
        """
        Helper function returning mask of batch shape (batch_size, max_time_step,) containing True at coords.
        Parameters
        ----------
        coords : numpy.array of shape (2, ?) of int
            Coordinates where mask should be True
        Returns
        -------
        mask : numpy.array of shape (batch_size, max_time_step,) of bool
            Matrix of shape = batch shape.
        """
        mask = np.full(shape=self.shape, fill_value=False, dtype=bool)
        mask[tuple(coords)] = True
        return mask

    '''update prefix_str'''

    def update_prefix_str(self, new_tokens_idx):
        mask_not_end = new_tokens_idx != self.all_token_info_table.invalid_id - 1  # end_id
        # prefix_str_add = np.full(self.shape[0], fill_value="", dtype='<U32')
        prefix_str_add = np.char.add(
            self.all_token_info_table.token_name_table[new_tokens_idx][mask_not_end],
            np.full(sum(mask_not_end), fill_value=","))
        self.prefix_str[:new_tokens_idx.shape[0]][mask_not_end] = np.char.add(
            self.prefix_str[:new_tokens_idx.shape[0]][mask_not_end], prefix_str_add)

    '''update_tokens_idx'''

    def update_new_tokens_idx(self, new_tokens_idx):
        self.new_tokens_idx = new_tokens_idx

    '''update_include_free_const'''

    def update_include_free_const(self):
        self.include_free_const = np.logical_or(self.include_free_const,
                                                self.all_token_info_table.type_table[self.new_tokens_idx] ==
                                                TYPE_KIND_NUMBER["free_const"])
    '''update_have_end'''

    def update_have_end(self):
        mask_end = self.all_token_info_table.type_table[self.new_tokens_idx] == TYPE_KIND_NUMBER["end"]
        self.have_end = np.logical_or(self.have_end, mask_end)
    
    '''update_combination_info'''

    def update_have_combination(self):
        mask_combination = self.all_token_info_table.type_table[self.new_tokens_idx] == TYPE_KIND_NUMBER["combination"]
        self.combination_info.have_combination = np.logical_or(self.combination_info.have_combination, mask_combination)
        if mask_combination.sum() > 0:
            self.combination_info.combination_tokens_id[mask_combination, self.current_time_step] = self.new_tokens_idx[
                mask_combination]
    
    '''update_real_length'''

    def update_real_length(self):
        self.real_length += self.all_token_info_table.length_table[self.new_tokens_idx]

    '''update one_hot of token: no use in this version'''

    def update_tokens_one_hot(self):
        start_index = self.current_time_step * self.tokens_number
        self.tokens_one_hot[:, start_index:start_index + self.tokens_number] = self.eye[self.new_tokens_idx]

    '''update is_physical'''

    def units_consistency(self, ):
        # 查看new_token的单位是否符合该token位置的要求（current_time_step已经在之前一次+1了）
        units_need = self.phy_units[:, self.current_time_step]
        units_new = self.all_token_info_table.phy_units_table[self.new_tokens_idx]
        units_unequal = np.logical_not((units_need == units_new).all(axis=1))  # all judge——if all are true in a prog

        # to avoid nan-nan ops equal
        mask_constraining_need = self.is_constraining_phy_units[:, self.current_time_step]
        self.mask_constraining_new = self.all_token_info_table.is_constraining_phy_units_table[
            self.new_tokens_idx]  # only constrain items, should be applied to update_units_state()
        # 查看哪些需要添加new_token的位置被单位约束，并且新加的new_token正好也被单位约束，并且这个new_token不满足这个单位约束，就记为units_inconsistency
        units_inconsistency = np.logical_and.reduce((  # (batch_size,) of bool
            units_unequal,
            mask_constraining_need,
            self.mask_constraining_new,
        ))
        self.units_inconsistency[np.where(units_inconsistency == True)[0]] = self.current_time_step #self.units_inconsistency是一个记录的变量。记录了1000个表达式中，哪些位置的单位不符合要求
        return units_inconsistency

    def update_is_physical(self):
        units_inconsistency = self.units_consistency()
        # if it was necessary to analysis units(if completed, don't need to check physical)
        necessary_units_analysis = np.logical_or(
            ~(self.units_analysis_cases[:, self.current_time_step, ] == UNITS_ANALYSIS_NOT_PERFORMED_CASE_CODE),
            self.have_completed)
        # 是否已经超出最大长度:0216
        # self.exceed_to_boundary()
        # if unnecessary==false, then self.is_physical & (~units_inconsistency)
        self.is_physical = self.is_physical & (~units_inconsistency) & necessary_units_analysis
        # print("number of functions meeting units constraints: ", sum(self.is_physical == True))

    '''update the phi_units state now'''
    # 需要这步吗？需要，因为虽然有些：对于这些位置，如果这个位置的new_token是constraining的，那么这个位置的phy_units就是这个new_token的phy_units.
    # 因为可能原来这里不是constraining的，但是新的token是constraining的，所以这个位置的phy_units就是这个new_token的phy_units。
    def update_units_state(self):
        self.is_constraining_phy_units[self.mask_constraining_new, self.current_time_step] = \
            self.all_token_info_table.is_constraining_phy_units_table[
                self.new_tokens_idx][self.mask_constraining_new] #self.is_constraining_phy_units是（1000，40）的矩阵，记录了1000个表达式中，哪些位置是constraining的
        self.phy_units[self.mask_constraining_new, self.current_time_step, :] = \
            self.all_token_info_table.phy_units_table[self.new_tokens_idx][self.mask_constraining_new]

    '''update_n_placeholders'''
    def update_n_placeholders(self):
        '''获得本轮加入new_token后，上一轮的placeholders还遗留多少'''
        # Number of dummies that will be left after new tokens replace
        # 1st dummy (counting all dummies minus dummy being replaced)
        # self.n_placeholder_left是上一轮的placeholder的数量-1:代表上一轮的placeholder有多少留到了这一轮。因为已经填上了一个token
        self.n_placeholder_left = self.n_placeholder_now - np.ones(self.shape[0], dtype=int)  # (batch_size,) of int
        # Complete programs do not need dummies
        self.n_placeholder_left[self.have_completed] = 0  # (self.have_completed.sum(),) of int

        '''获得本轮加入new_token后，一共多少placeholder'''
        # Update program lengths for those that are still incomplete
        self.n_lengths[self.have_completed == False] += 1  # (self.have_completed.sum(),) of int
        # Update of arities over time dim with new tokens
        self.total_arities += self.all_token_info_table.arity_table[self.new_tokens_idx]  # (batch_size,) of int
        # Number of dummy placeholders
        self.n_placeholder_now = self.total_arities - self.n_lengths  # (batch_size,) of int——这里进行了n_placeholder_now的更新
        
    '''update_length_unphysical:由于有时所有符号都被prior认为不能取，导致它随便取了一个符号，相当于没有考虑到Lengthprior不允许self.real_length突破self.max_time_step的约束。因此需要在这里重新进行一次判定，判定失败则self.is_physical为False'''
    def update_length_unphysical(self):
        mask_exceed_length = (self.real_length + self.n_placeholder_now) > self.shape[1]
        self.is_physical[mask_exceed_length] = False
    
    '''update_have_completed'''
    def update_have_completed(self):
        # update have_completed：并且要在self.have_end==False时判断，因为存在sin x sin end cos ...的情况
        self.have_completed[
            np.logical_and.reduce((self.have_completed == False,
                           self.n_placeholder_now == 0,
                           self.have_end==False))] = True  # 当你直接传入三个条件给 np.logical_and 时，会出错。原因是 np.logical_and 函数只接受两个参数。
        
    '''update_complexity'''

    def update_complexity(self):
        # Update of arities over time dim with new tokens
        self.total_complexity += self.all_token_info_table.complexity_table[self.new_tokens_idx]  # (batch_size,) of int

    '''update_previous_exsiting_placeholders_relations——通过shift关系'''

    def need_shift_placeholders(self):
        '''先确定哪些先前的placeholder需要被移动；'''
        batch_pos = np.tile(np.arange(0, self.shape[1]), (self.shape[0], 1)).astype(int)
        # 上一轮还遗留的placeholder（已经减过1）的位置
        mask_placeholder_left = np.logical_and.reduce((batch_pos >= self.current_time_step+1,
                                                       batch_pos < self.current_time_step+1 + np.tile(
                                                           self.n_placeholder_left,
                                                           (self.max_time_step, 1)).transpose()))
        # 是否需要移动placeholder
        mask_batch_have_placeholder_add = (self.n_placeholder_need_add > 0) & self.is_physical # 已经不物理约束的（尤其是上述的length超过的）,就不进行移动了
        mask_batch_have_placeholder_add = np.tile(mask_batch_have_placeholder_add, (self.max_time_step, 1)).transpose()
        
        # 哪些位置的placeholder需要被移动
        mask_placeholder_need_shift = np.logical_and(mask_placeholder_left, mask_batch_have_placeholder_add)

        # Coords of legacy tokens
        # shift之前的placeholder的位置
        n_placeholders_before_shift, coords_placeholders_before_shift = self.mask_to_coords(
            mask_placeholder_need_shift)
        # shift之后的placeholder的位置
        coords_placeholders_after_shift = np.stack((  # (2, n_legacy_dummies_total,) of int
            coords_placeholders_before_shift[0],  # batch dim coord -> no change
            coords_placeholders_before_shift[1] + self.n_placeholder_need_add[coords_placeholders_before_shift[0]],
        ), axis=0)
        return coords_placeholders_before_shift, coords_placeholders_after_shift

    def shift_before_placeholders_de_relation(self, coords_placeholders_before_shift, coords_placeholders_after_shift,
                                    insert_position=None, shift_length=0):
        '''把before placeholder的信息，传递给after；并且反过来把（1）after的父节点的子节点index，更新为after；（2）after的兄弟节点的兄弟节点index，更新为after'''
        def shift_after_placeholders_parent_de_children(coords_placeholders_before_shift, coords_placeholders_after_shift):
            parent_index = self.parent_index[
                tuple(
                    coords_placeholders_after_shift)]  # the parent index of whom have parent: need to be after shift(or will be error when meet 1-op)
            mask_have_parent = ~(parent_index >= ILLEGAL_RELATIVE_INDEX)  # who have parent

            parent_coords_need_to_shift_children = np.stack((coords_placeholders_after_shift[0], parent_index), axis=0)
            valid_parent_coords_need_to_shift_children = parent_coords_need_to_shift_children[:,
                                                         mask_have_parent]  # valid parent coords before shift

            children_equal_sign = np.tile(coords_placeholders_before_shift[1], (2, 1)).transpose()
            children_equal_sign = children_equal_sign[mask_have_parent]

            children_index_before_shift = self.children_index[tuple(valid_parent_coords_need_to_shift_children)]
            children_index_before_shift[children_index_before_shift == children_equal_sign] = \
                coords_placeholders_after_shift[1][
                    mask_have_parent]
            children_index_after_shift = children_index_before_shift
            self.children_index[tuple(valid_parent_coords_need_to_shift_children)] = children_index_after_shift

            '''replace by get_children_end()'''
            # children_end_before_shift = self.children_end[tuple(valid_parent_coords_need_to_shift_children)]
            # children_end_before_shift[children_end_before_shift == children_equal_sign] += \
            #     (coords_placeholders_after_shift[1][mask_have_parent] - coords_placeholders_before_shift[1][
            #         mask_have_parent])[mask_have_parent]
            # children_end_after_shift = children_end_before_shift
            # self.children_end[tuple(valid_parent_coords_need_to_shift_children)] = children_end_after_shift

        def shift_after_placeholders_sibling_de_sibling(coords_placeholders_before_shift, coords_placeholders_after_shift):
            sibling_index = self.sibling_index[
                tuple(coords_placeholders_before_shift)]  # the parent index of whom have parent
            mask_have_sibling = ~(sibling_index >= ILLEGAL_RELATIVE_INDEX)  # who have parent
            sibling_coords_need_to_shift_sibling = np.stack((coords_placeholders_before_shift[0], sibling_index),
                                                            axis=0)
            valid_sibling_coords_need_to_shift_sibling = sibling_coords_need_to_shift_sibling[:,
                                                         mask_have_sibling]  # valid sibling coords before shift

            sibling_before_shift = self.sibling_index[tuple(valid_sibling_coords_need_to_shift_sibling)]
            sibling_after_shift = coords_placeholders_after_shift[1][mask_have_sibling]
            self.sibling_index[tuple(valid_sibling_coords_need_to_shift_sibling)] = sibling_after_shift
        # 首先把需要shift的placeholder的信息，传递给新的位置的信息
        self.is_constraining_phy_units[tuple(coords_placeholders_after_shift)] = self.is_constraining_phy_units[
            tuple(coords_placeholders_before_shift)]
        self.phy_units[tuple(coords_placeholders_after_shift)] = self.phy_units[tuple(coords_placeholders_before_shift)]
        self.height[tuple(coords_placeholders_after_shift)] = self.height[
            tuple(coords_placeholders_before_shift)]
        self.children_index[tuple(coords_placeholders_after_shift)] = self.children_index[
                                                                          tuple(
                                                                              coords_placeholders_before_shift)] + shift_length  # all children will be after insert place
        self.children_number[tuple(coords_placeholders_after_shift)] = self.children_number[
            tuple(coords_placeholders_before_shift)]
        # self.children_end[tuple(coords_placeholders_after_shift)] = self.children_end[
        #                                                                 tuple(
        #                                                                     coords_placeholders_before_shift)] + shift_length  # all children will be after insert place

        # ---- relative info, but need to consider if the index before or after insert position----
        if insert_position is None:
            self.parent_index[tuple(coords_placeholders_after_shift)] = self.parent_index[
                tuple(coords_placeholders_before_shift)]
            self.sibling_index[tuple(coords_placeholders_after_shift)] = self.sibling_index[
                tuple(coords_placeholders_before_shift)]
            shift_after_placeholders_parent_de_children(coords_placeholders_before_shift, coords_placeholders_after_shift) # 齐次，反过来把after的父节点的子节点index，更新为after
            shift_after_placeholders_sibling_de_sibling(coords_placeholders_before_shift, coords_placeholders_after_shift) # 最后，反过来把after的兄弟节点的兄弟节点index，更新为after

        else:
            origin_parent_index = self.parent_index[tuple(coords_placeholders_before_shift)]
            mask_parent_before_insert = origin_parent_index <= insert_position
            mask_parent_after_insert = origin_parent_index > insert_position
            self.parent_index[tuple(coords_placeholders_after_shift[:, mask_parent_before_insert])] = \
                origin_parent_index[mask_parent_before_insert]
            self.parent_index[tuple(coords_placeholders_after_shift[:, mask_parent_after_insert])] = \
                origin_parent_index[mask_parent_after_insert] + shift_length
            if mask_parent_before_insert.sum() > 0:
                shift_after_placeholders_parent_de_children(coords_placeholders_before_shift[:, mask_parent_before_insert],
                                                   coords_placeholders_after_shift[:, mask_parent_before_insert])

            # insert_coords = np.array(
            #     [coords_placeholders_before_shift[0, 0], insert_position]).reshape((2, 1))
            # coords_need_to_consider_sibling = np.hstack(
            #     (insert_coords, coords_placeholders_before_shift))  # for the column stack up
            # coords_need_to_consider_sibling_after=np.hstack(
            #     (insert_coords, coords_placeholders_after_shift))  # for the column stack up
            origin_sibling_index = self.sibling_index[tuple(coords_placeholders_before_shift)]
            mask_sibling_before_insert = origin_sibling_index <= insert_position
            mask_sibling_after_insert = origin_sibling_index > insert_position
            self.sibling_index[tuple(coords_placeholders_after_shift[:, mask_sibling_before_insert])] = \
                origin_sibling_index[mask_sibling_before_insert]
            self.sibling_index[tuple(coords_placeholders_after_shift[:, mask_sibling_after_insert])] = \
                origin_sibling_index[mask_sibling_after_insert] + shift_length
            if mask_sibling_before_insert.sum() > 0:
                shift_after_placeholders_sibling_de_sibling(coords_placeholders_before_shift[:, mask_sibling_before_insert],
                                           coords_placeholders_after_shift[:, mask_sibling_before_insert])

        # self.depth[tuple(coords_placeholders_after_shift)] = self.depth[tuple(coords_placeholders_before_shift)]
        # self.children_end[tuple(coords_placeholders_after_shift)] = self.children_end[
        #     tuple(coords_placeholders_before_shift)]
    
    def update_previous_exsiting_placeholders_relations(self):
        # Complete programs do not need dummies
        self.n_placeholder_now[self.have_completed] = 0  # (self.have_completed.sum(),) of int
        '''self.n_placeholder_need_add代表这一轮各个表达式需要添加的placeholder的数量'''
        self.n_placeholder_need_add = (self.n_placeholder_now - self.n_placeholder_left).astype(int) # self.n_placeholder_left是上一time step且加了new_token后，剩下的placeholder；self.n_placeholder_now是这一time step的placeholder; self.n_placeholder_need_add是这一time step需要添加的placeholder
        if sum(self.n_placeholder_left) > 0: # 上一step之后，填上这个token以后，还留下的placeholder
            coords_placeholders_before_shift,coords_placeholders_after_shift=self.need_shift_placeholders()  # 已经不物理约束的（尤其是上述的length超过的）,就不进行移动了。主要看n_placeholder_need_add
            self.shift_before_placeholders_de_relation(coords_placeholders_before_shift,coords_placeholders_after_shift)  # tuple tuple

    '''update的utils函数：set parent, children, sibling index'''

    def set_parent(self, coords, parent_pos):
        self.parent_index[coords[0], coords[1]] = parent_pos

    def set_children(self, coords, child_pos, child_nb=1):
        self.children_index[coords[0], coords[1]] = child_pos
        self.children_number[coords[0], coords[1]] = child_nb

    def set_sibling(self, coords, sibling_pos):
        self.sibling_index[coords[0], coords[1]] = sibling_pos

    '''update add placeholders relations'''
    def get_new_placeholder_coords(self): # 在加入new_token后，所产生的新placeholder位置
        """
        Helper function returning coordinates where mask is True.
        Parameters
        ----------

        Returns
        -------
        mask_sum, coordinates : int, numpy.array of shape (2, mask_sum) of int
            Number of coordinates where mask is True, coordinates where mask is True.
        """
        # Showing that memory space can accurately be allocated before-hand
        self.n_placeholder_need_add = self.n_placeholder_need_add.astype(int)
        self.n_placeholder_need_add = np.where(self.n_placeholder_need_add < 0, 0, self.n_placeholder_need_add)
        self.n_placeholder_need_add[self.is_physical == False] = 0 # 不满足物理约束的，就不进行add placeholder了0217
        mask_sum = self.n_placeholder_need_add.sum()  # int
        coords = np.zeros(shape=(2, mask_sum,), dtype=int)  # (2, mask_sum,) of int
        count = 0
        # Coordinates
        for batch in range(len(self.n_placeholder_need_add)):
            for add_placeholder in range(1, self.n_placeholder_need_add[batch] + 1, 1):
                coords[0, count] = batch
                coords[1, count] = add_placeholder + self.current_time_step
                count += 1
        return mask_sum, coords

    '''update relations的内部函数,用于更新add placeholder父节点的children，为add placeholder'''
    def update_children(self, coords_lonely_child, coords_sibling0, coords_sibling1):
        arity_new = np.tile(self.all_token_info_table.arity_table[self.new_tokens_idx],
                            (self.max_time_step, 1)).transpose()
        mask_new_tokens = np.full(shape=self.shape, fill_value=False, dtype=bool)
        mask_new_tokens[:, self.current_time_step] = True  # which time step add new token
        mask_incomplete = np.logical_not(np.tile(self.have_completed, (self.max_time_step, 1)).transpose())
        mask_inphysical = np.tile(self.is_physical, (self.max_time_step, 1)).transpose() #0216
        # mask_incomplete=mask_incomplete & mask_inphysical
        # -------- HANDLING LONELY CHILDREN --------
        # New tokens having only one dummy child : mask
        # Ie. is a new token AND has arity == 1 AND program is not complete
        # 本质上就是加入new_token的位置
        mask_new_tokens_w_lonely_child = np.logical_and.reduce((  # (batch_size, max_time_step,) of bool
            mask_new_tokens,
            arity_new == 1,
            mask_incomplete,
            mask_inphysical
        ))
        
        # n_new_tokens_w_lonely_child = mask_new_tokens_w_lonely_child.sum()
        n_new_tokens_w_lonely_child, coords_new_tokens_w_lonely_child = self.mask_to_coords(
            mask_new_tokens_w_lonely_child)  # int, (2, n_new_tokens_w_lonely_child,) of int
        
        # Positions of lonely children in time dim
        pos_lonely_children = np.stack((  # (n_new_tokens_w_lonely_child, 1,) of int
            coords_lonely_child[1],  # position of child 0 in time dim
            np.full(n_new_tokens_w_lonely_child, ILLEGAL_RELATIVE_INDEX, int)),  # no 2nd child
            axis=1, )
        # Setting children：设置只有一个孩子的new_tokens的children为，coords_lonely_child和ILLIGAL_RELATIVE_INDEX(代表第二个孩子为空)
        self.set_children(coords=tuple(coords_new_tokens_w_lonely_child),
                          # (2, n_new_tokens_w_lonely_child,) of int
                          child_pos=pos_lonely_children,  # (n_new_tokens_w_lonely_child, 1,) of int
                          child_nb=1,  # (n_new_tokens_w_lonely_child,) of int
                          )

        # -------- HANDLING DOUBLE CHILDREN --------
        # New tokens having two children : mask
        # Ie. is a new token AND has arity == 2 AND program is not complete
        mask_new_tokens_w_two_children = np.logical_and.reduce((  # (batch_size, max_time_step,) of bool
            mask_new_tokens,
            arity_new == 2,
            mask_incomplete,
            mask_inphysical
        ))
        # n_new_tokens_w_two_children = mask_new_tokens_w_two_children.sum()
        n_new_tokens_w_two_children, coords_new_tokens_w_two_children = self.mask_to_coords(
            mask_new_tokens_w_two_children)  # int, (2, n_new_tokens_w_two_children,) of int
        # Positions of double children in time dim
        pos_double_children = np.stack((  # (n_new_tokens_w_two_children, 2,) of int
            coords_sibling0[1],  # position of child 0 in time dim
            coords_sibling1[1]),  # position of child 1 in time dim
            axis=1)
        
        # Setting children：设置两个孩子的new_tokens的children为，coords_sibling0和coords_sibling1,
        self.set_children(coords=tuple(coords_new_tokens_w_two_children),
                          # (2, n_new_tokens_w_two_children,) of int
                          child_pos=pos_double_children,  # (n_new_tokens_w_two_children, 2,) of int
                          child_nb=2,  # (n_new_tokens_w_two_children,) of int
                          )
    
    def update_add_placeholders_relations(self):
        '''更新新加入token后，add placeholder位置的parent, sibling; 反过来把add placeholder的parent的children更新为add placeholder'''
        
        '''首先得到加入new_token后，所产生的新placeholder位置 self.add_placeholder_coords'''
        # 为了防止后续shift relation的错误，self.is_physical==False的地方，就不进行self.n_placeholder_need_add:0216
        # self.n_placeholder_need_add[self.is_physical == False] = 0
        _, self.add_placeholder_coords = self.get_new_placeholder_coords()  # after op, we need 1 or 2 placeholders to place its children
        
        batch_id = np.arange(start=0, stop=self.batch_size, step=1)
        # update depth
        # self.set_depth(coords=self.add_placeholder_coords, parent_pos=self.current_time_step)

        '''更新add_placeholder_coords的parent'''
        # update parent：new_token后面新加入的placeholder的parent是new_token
        self.set_parent(coords=self.add_placeholder_coords, parent_pos=self.current_time_step)

        # update sibling
        # mask_have_no_sibling = self.n_placeholder_need_add <= 1
        # coords_sibling = np.stack((batch_id[mask_have_no_sibling == True], np.full(shape=sum(mask_have_no_sibling),
        #                                                                            fill_value=self.current_time_step,
        #                                                                            dtype=int)), axis=0)
        # self.sibling_index[coords_sibling] = ILLEGAL_RELATIVE_INDEX  # to invalid

        '''如果新加入token是arity==2的，就说明要有两个新placeholder加入，那么就有sibling0和sibling1'''
        mask_have_sibling = self.n_placeholder_need_add == 2
        mask_have_sibling=self.is_physical & mask_have_sibling # 不满足物理约束的，就不进行sibling的更新
        coords_sibling0 = np.stack((batch_id[mask_have_sibling == True], np.full(shape=sum(mask_have_sibling),
                                                                                 fill_value=self.current_time_step+1,
                                                                                 dtype=int)), axis=0) # 新加入placeholder1
        coords_sibling1 = np.stack((batch_id[mask_have_sibling == True], np.full(shape=sum(mask_have_sibling),
                                                                                 fill_value=self.current_time_step + 2,
                                                                                 dtype=int)), axis=0) #新加入placeholder2
        '''更新add_placeholder_coords的sibling'''
        # 将两个新加入placeholder彼此的sibling设置为对方
        self.set_sibling(coords=coords_sibling0, sibling_pos=self.current_time_step + 2)
        self.set_sibling(coords=coords_sibling1, sibling_pos=self.current_time_step+1)

        
        '''如果新加入token是arity==1的，就说明要有一个新placeholder加入，那么就有coords_lonely_child'''
        # update children：更新新加入的placeholder的children
        mask_have_lonely_child = self.n_placeholder_need_add == 1 #加入的符号是单节点的
        mask_have_lonely_child = self.is_physical & mask_have_lonely_child # 不满足物理约束的，就不进行children的更新
        # mask_inphysical = self.is_physical #0216新增
        # mask_have_lonely_child = mask_have_lonely_child & mask_inphysical
        coords_lonely_child = np.stack(
            (batch_id[mask_have_lonely_child == True], np.full(shape=sum(mask_have_lonely_child),
                                                               fill_value=self.current_time_step+1,
                                                               dtype=int)), axis=0) # 新加入的placeholder的位置（也就是这个新增单空白的位置）
        
        '''对于add_placeholder_coords的父节点new_token，更新它的children'''
        self.update_children(coords_lonely_child, coords_sibling0, coords_sibling1)

    # def set_depth(self, coords, parent_pos):
    #     coords_new_placeholders_parents = np.stack((  # (2, n_new_dummies_total,) of int
    #         coords[0],  # batch dim coord
    #         np.full(len(coords[0]), parent_pos),  # time dim coord
    #         # = self.tokens.pos[coords_new_dummies[0], self.curr_step - 1]
    #     ), axis=0)
    #     self.depth[coords[0], coords[1]] = self.depth[
    #                                            coords_new_placeholders_parents[0], coords_new_placeholders_parents[
    #                                                1]] + 1  # (n_new_dummies_total,) of int

    '''get属性所需函数'''
    '''get parent and sibling index and token_id'''

    def get_parent_idx(self, time_step, coords):  # (batch_size)
        parent_index = self.parent_index[:, time_step]
        parent_idx = np.full(shape=self.batch_size,
                             fill_value=ILLEGAL_RELATIVE_INDEX,
                             dtype=int)
        mask_have_parent = ~(self.parent_index[:, time_step] >= ILLEGAL_RELATIVE_INDEX)

        parent_idx[mask_have_parent] = self.tokens_idx[mask_have_parent, parent_index[mask_have_parent]]
        # update legal parent idx——for items which have parent
        return mask_have_parent[coords[0]], parent_index[coords[0]], parent_idx[coords[0]]  # get coords's parent
    
    def get_parent_index_of_progi(self, prog_id, time_step):
        parent_index = self.parent_index[prog_id, time_step]
        return parent_index
    
    def get_ancestor_idx_of_progi(self, prog_id, origin_time_step):
        ancestors = []
        def get_ancestor_recursive(time_step):
            parent_index = self.get_parent_index_of_progi(prog_id,time_step)
            if parent_index >= ILLEGAL_RELATIVE_INDEX:
                return
            ancestors.append(parent_index)
            get_ancestor_recursive(parent_index)

        get_ancestor_recursive(origin_time_step)
        ancestors_idx=[self.tokens_idx[prog_id, ancestor] for ancestor in ancestors] # get token_id
        return ancestors_idx

    def get_sibling_idx(self, time_step, coords):
        sibling_index = self.sibling_index[:, time_step]  # index
        sibling_idx = np.full(shape=self.batch_size,
                              fill_value=ILLEGAL_RELATIVE_INDEX,
                              dtype=int)  # token_id
        mask_have_sibling = ~(self.sibling_index[:, time_step] >= ILLEGAL_RELATIVE_INDEX)
        sibling_not_empty = np.full(shape=self.batch_size,
                                    fill_value=False,
                                    dtype=bool)
        sibling_not_empty[mask_have_sibling] = ~(self.tokens_idx[mask_have_sibling, sibling_index[
            mask_have_sibling]] == self.all_token_info_table.invalid_id)  # (have_sibling--->row, sibling_index--->col)
        sibling_idx[sibling_not_empty] = self.tokens_idx[
            sibling_not_empty, sibling_index[sibling_not_empty]]
        return mask_have_sibling[coords[0]], sibling_not_empty[coords[0]], sibling_index[coords[0]], sibling_idx[
            coords[0]]  # get legal index

    '''get height'''

    def single_batch_height(self, batch_id, index):
        if self.height[batch_id, index] != 0:
            return self.height[batch_id, index]
        elif self.tokens_idx[batch_id, index] in self.all_token_info_table.combination_group:
            self.height[batch_id, index] = self.combination_info.height_table[self.tokens_idx[batch_id, index]][0]
            return self.height[batch_id, index]
        else:
            if self.mask_have_one_child[batch_id, index] == True:
                self.height[batch_id, index] = self.single_batch_height(batch_id,
                                                                        self.children_index[batch_id, index][0]) + 1
                return self.height[batch_id, index]
            elif self.mask_have_two_children[batch_id, index] == True:
                self.height[batch_id, index] = max(
                    self.single_batch_height(batch_id, self.children_index[batch_id, index][0]),
                    self.single_batch_height(batch_id, self.children_index[batch_id, index][1])) + 1
                return self.height[batch_id, index]
            else:
                self.height[batch_id, index] = 1
                return self.height[batch_id, index]

    def get_height(self):
        self.mask_have_one_child = self.children_number == 1
        self.mask_have_two_children = self.children_number == 2
        for batch_id in range(self.batch_size):
            if self.is_physical[batch_id]:  # only update those are physical, because they will be computed similarity
                for index in range(self.n_lengths[batch_id]):
                    self.single_batch_height(batch_id, index)

    '''get children_end'''

    def prog_index_children_end(self, batch_id, index):
        if self.mask_have_one_child[batch_id, index] == True:
            self.children_end[batch_id, index][0] = self.prog_index_children_end(batch_id,
                                                                                 self.children_index[batch_id, index][
                                                                                     0])
            return self.children_end[batch_id, index][0]
        elif self.mask_have_two_children[batch_id, index] == True:
            self.children_end[batch_id, index][0] = self.children_index[batch_id, index][1] - 1
            self.children_end[batch_id, index][1] = self.prog_index_children_end(batch_id,
                                                                                 self.children_index[batch_id, index][
                                                                                     1])
            return self.children_end[batch_id, index][1]
        else:
            return index

    def get_children_end_and_length(self):
        self.mask_have_one_child = self.children_number == 1
        self.mask_have_two_children = self.children_number == 2
        for batch_id in range(self.batch_size):
            if self.is_physical[batch_id]:  # only update those are physical, because they will be computed similarity
                mask_height_above1 = self.height[batch_id] > 1
                for index in np.where(mask_height_above1)[0]:  # only update those with height above 1
                    self.prog_index_children_end(batch_id, index)
                    self.children_length[batch_id, index] = self.children_end[batch_id, index] - self.children_index[
                        batch_id, index] + 1

    '''get children_end of single prog_id for similarity calculation'''

    def get_prog_children_end_and_length(self, prog_id):
        mask_height_above1 = self.height[prog_id] > 1
        for index in np.where(mask_height_above1)[0]:  # only update those with height above 1
            self.prog_index_children_end(prog_id, index)
            self.children_length[prog_id, index] = self.children_end[prog_id, index] - self.children_index[
                prog_id, index] + 1

    '''similarity_calculation need'''
    '''update relations and height for similarity calculation'''

    def update_height_children(self, combination_id, prog_id, combination_index, have_shift_length=0):
        batch_pos = np.tile(np.arange(0, self.shape[1]), (1, 1)).astype(int)
        shift_length = self.all_token_info_table.length_table[combination_id] - 1
        mask_placeholder_need_to_shift = np.logical_and.reduce((batch_pos >= combination_index + 1,
                                                                batch_pos < self.n_lengths[
                                                                    prog_id] + have_shift_length))  #
        mask_sum = mask_placeholder_need_to_shift.sum()  # int
        if mask_sum > 0:
            coords_placeholders_before_shift = np.zeros(shape=(2, mask_sum,), dtype=int)  # (2, mask_sum,) of int
            # Coordinates
            coords_placeholders_before_shift[:, :] = np.stack((  # (2, mask_sum,) of int
                np.full(mask_sum, fill_value=prog_id, dtype=int),  # batch dim coord
                np.where(mask_placeholder_need_to_shift)[1],  # time dim coord
            ), axis=0)
            # n_placeholders_before_shift, coords_placeholders_before_shift = self.mask_to_coords(
            #     mask_placeholder_need_to_shift)
            coords_placeholders_after_shift = np.stack((  # (2, n_legacy_dummies_total,) of int
                coords_placeholders_before_shift[0],  # batch dim coord -> no change
                coords_placeholders_before_shift[1] + shift_length,
            ), axis=0)
            self.shift_before_placeholders_de_relation(coords_placeholders_before_shift,
                                             coords_placeholders_after_shift, insert_position=combination_index,
                                             shift_length=shift_length)  # tuple tuple
        self.height[prog_id, combination_index:combination_index + shift_length + 1] = \
            np.array(self.combination_info.height_table[combination_id])
        self.parent_index[prog_id, combination_index + 1:combination_index + shift_length + 1] = \
            np.array(self.combination_info.parent_index_table[combination_id][1:]) + combination_index
        self.sibling_index[prog_id, combination_index + 1:combination_index + shift_length + 1] = \
            np.array(self.combination_info.sibling_index_table[combination_id][1:]) + combination_index
        self.children_index[prog_id, combination_index:combination_index + shift_length + 1] = \
            np.array(self.combination_info.children_index_table[combination_id]) + combination_index
        self.children_number[prog_id, combination_index:combination_index + shift_length + 1] = \
            np.array(self.combination_info.children_number_table[combination_id])
        # self.children_end[prog_id, combination_index:combination_index + shift_length + 1] = \
        #     np.array(self.combination_info.children_end_table[combination_id]) + combination_index

    def update_combination_height_children(self, prog_id):
        have_shift_length = 0
        for index, combination_id in enumerate(self.combination_info.combination_tokens_id[prog_id]):
            if combination_id != ILLEGAL_RELATIVE_INDEX:
                self.update_height_children(combination_id, prog_id, combination_index=index + have_shift_length,
                                            have_shift_length=have_shift_length)
                have_shift_length += self.all_token_info_table.length_table[combination_id] - 1
            if index + have_shift_length >= self.real_length[prog_id] - 1:  # >=last index
                break
        self.mask_have_one_child[prog_id] = self.children_number[prog_id] == 1
        self.mask_have_two_children[prog_id] = self.children_number[prog_id] == 2
        
    '''post_physical_check_of_occupancy'''

    '''update llm info(about templates): no use in this version'''

    def update_use_llm(self, prog_id_sample: np.array, current_time_step: int, phy_units: str):
        probabilities = self.llm_info.llm.templates_probabilities[phy_units]
        samples = np.random.choice(len(probabilities), size=len(prog_id_sample),
                                   p=probabilities)  # 从数字中按照概率进行随机抽取id_sample个数字
        length_samples = np.array(self.llm_info.llm.templates_length[phy_units])[samples]
        n_placeholder_now = self.n_placeholder_now[prog_id_sample]
        mask_samples_valid = length_samples + n_placeholder_now - 1 + current_time_step < self.max_time_step  # last place is end

        for i, prog_id in enumerate(prog_id_sample):
            if mask_samples_valid[i]:
                self.llm_info.use_llm[prog_id, current_time_step:current_time_step + length_samples[i]] = True
                self.llm_info.llm_tokens_name[prog_id, current_time_step:current_time_step + length_samples[i]] = \
                    self.llm_info.llm.templates[phy_units][samples[i]].tokens_name
                self.llm_info.tokens_id[prog_id, current_time_step:current_time_step + length_samples[i]] = \
                    self.llm_info.llm.templates[phy_units][samples[i]].tokens_id
