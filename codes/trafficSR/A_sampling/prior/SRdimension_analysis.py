import numpy as np


def assign_required_units(programs, coords, ):
    # -------------------- Define variables -------------------- #DONE
    # Positions
    batch_pos = coords[0, :]  # Position in batch dim                                                      # (n_tokens,)
    pos = coords[1, :]  # Position in time sequence                                                  # (n_tokens,)
    n_tokens = len(batch_pos)
    current_time_step = pos[0]

    case_code = 0
    situation_code_recode = np.zeros(n_tokens, dtype=int)

    # units constraint
    # mask : do we have constraints regarding the physical units of the token ?
    # By default = False (no constraint)
    is_constraining = np.full(shape=n_tokens, fill_value=False, dtype=bool)  # (n_tokens,)

    # Required units for each token in batch. By default = np.nan (no constraint)
    phy_units = np.full(shape=(n_tokens, 7), fill_value=np.nan,
                        dtype=float)  # (n_tokens, UNITS_VECTOR_SIZE)

    # -------------------- Define constraint functions -------------------- #DONE
    def apply_constraints(mask_case, inferred_is_constraining, inferred_phy_units):
        is_constraining[mask_case] = inferred_is_constraining
        phy_units[mask_case, :] = inferred_phy_units[mask_case]

    def apply_const_constraints(mask_case, const_is_constraining, const_phy_units):
        is_constraining[mask_case] = const_is_constraining
        phy_units[mask_case, :] = const_phy_units

    # -------------------- CASE 0 --------------------
    # if it is the first step, apply superparent to initialize
    if current_time_step == 0:
        case_code = -1
        mask_case = situation_code_recode == 0
        situation_code_recode[mask_case] = case_code
        apply_const_constraints(mask_case, True, programs.library.super_parent_info.superparent_units)

    # -------------------- get relative info --------------------
    # if it is the first step, apply superparent
    mask_have_parent, parent_index, parent_idx = programs.library.get_parent_idx(time_step=current_time_step,
                                                                                 coords=coords)
    mask_have_sibling, sibling_not_empty, sibling_index, sibling_idx = programs.library.get_sibling_idx(
        time_step=current_time_step, coords=coords)

    # -------------------- Define relative functions -------------------- #DONE
    def get_parent_info():
        parent_is_constraining = np.full(shape=n_tokens, fill_value=False, dtype=bool)  # (n_tokens,)
        parent_phy_units = np.full(shape=(n_tokens, 7), fill_value=np.nan,
                                   dtype=float)  # (n_tokens,)
        parent_power = np.full(shape=n_tokens, fill_value=np.nan, dtype=float)
        parent_dimension_analysis_case = np.full(shape=n_tokens, fill_value=-1, dtype=int)  # (n_tokens,)

        parent_is_constraining[mask_have_parent] = programs.library.is_constraining_phy_units[
            batch_pos[mask_have_parent], parent_index[mask_have_parent]]
        parent_phy_units[mask_have_parent] = programs.library.phy_units[
            batch_pos[mask_have_parent], parent_index[mask_have_parent]]
        parent_power[mask_have_parent] = programs.library.all_token_info_table.power_table[parent_idx[mask_have_parent]]
        parent_dimension_analysis_case[mask_have_parent] = \
            programs.library.all_token_info_table.dimension_analysis_case_table[
                parent_idx[mask_have_parent]]
        return parent_is_constraining, parent_phy_units, parent_power, parent_dimension_analysis_case

    def get_sibling_info():
        sibling_is_constraining = np.full(shape=n_tokens, fill_value=False, dtype=bool)  # (n_tokens,)
        sibling_phy_units = np.full(shape=(n_tokens, 7), fill_value=np.nan,
                                    dtype=float)  # (n_tokens,)
        sibling_dimension_analysis_case = np.full(shape=n_tokens, fill_value=-1, dtype=int)  # (n_tokens,)

        sibling_is_constraining[sibling_not_empty] = programs.library.is_constraining_phy_units[
            batch_pos[sibling_not_empty], sibling_index[sibling_not_empty]]
        sibling_phy_units[sibling_not_empty] = programs.library.phy_units[
            batch_pos[sibling_not_empty], sibling_index[sibling_not_empty]]
        sibling_dimension_analysis_case[sibling_not_empty] = \
            programs.library.all_token_info_table.dimension_analysis_case_table[
                sibling_idx[sibling_not_empty]]
        return sibling_is_constraining, sibling_phy_units, sibling_dimension_analysis_case

    parent_is_constraining, parent_phy_units, parent_power, parent_dimension_analysis_case = get_parent_info()
    sibling_is_constraining, sibling_phy_units, sibling_dimension_analysis_case = get_sibling_info()

    # -------------------- CASE 1 --------------------
    # + -
    case_code = 1
    mask_case = np.logical_and.reduce(
        (situation_code_recode == 0, mask_have_parent, parent_dimension_analysis_case == 1))
    n_in_case = mask_case.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case] = case_code
        # print("case 1: ", n_in_case)
    # -------------------- CASE 10 --------------------
    # have sibling AND sibling not empty AND sibling constraint
    case_code = 10
    mask_case = np.logical_and.reduce((situation_code_recode == 1, mask_have_sibling,
                                       sibling_not_empty,
                                       sibling_is_constraining))  # more than 2 conditions, we need .reduce
    n_in_case = mask_case.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case] = case_code
        apply_constraints(mask_case, True, sibling_phy_units)
        # print("case 10: ", n_in_case)

    # -------------------- CASE 11 --------------------
    # parent constraint AND have sibling AND sibling empty
    case_code = 11
    mask_case = np.logical_and.reduce((situation_code_recode == 1, parent_is_constraining, mask_have_sibling,
                                       ~sibling_not_empty))
    n_in_case = mask_case.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case] = case_code
        apply_constraints(mask_case, True, parent_phy_units)  # get from parent units
        # print("case 11: ", n_in_case)

    # -------------------- CASE 12 --------------------
    # parent constraint AND have sibling AND sibling empty
    # mul add mul s v s0 (when input s0 step, it is error, because parent add not constraint)
    case_code = 12
    mask_case = np.logical_and.reduce(
        (situation_code_recode == 1, ~parent_is_constraining, sibling_not_empty, sibling_is_constraining == False))
    n_in_case = mask_case.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case] = case_code
        # Determining subtrees on which to perform bottom up dimensional analysis's start and ends
        # Start of subtree is the sibling of the current token
        coords_start = np.stack((batch_pos, sibling_index), axis=0)[:, mask_case]
        # End of subtree is the token just before the current token (pos - 1)
        coords_end = np.stack((batch_pos, pos - 1), axis=0)[:, mask_case]
        # Perform dimensional analysis
        assign_units_bottom_up(programs=programs, coords_start=coords_start, coords_end=coords_end)
        parent_is_constraining, parent_phy_units, parent_power, parent_dimension_analysis_case = get_parent_info()
        sibling_is_constraining, sibling_phy_units, sibling_dimension_analysis_case = get_sibling_info()
        apply_constraints(mask_case, True, sibling_phy_units)  # get from parent units
        # print("case 12: ", n_in_case)

    # -------------------- CASE 2 --------------------
    # */
    case_code = 2
    mask_case = np.logical_and.reduce((situation_code_recode == 0, mask_have_parent,
                                       np.logical_or(parent_dimension_analysis_case == 20,
                                                     parent_dimension_analysis_case == 21)))
    n_in_case = mask_case.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case] = case_code
        # print("case 2: ", n_in_case)

    # -------------------- CASE 20 --------------------
    # have sibling AND sibling empty
    case_code = 20
    mask_case = np.logical_and.reduce((situation_code_recode == 2, mask_have_sibling,
                                       ~sibling_not_empty))
    n_in_case = mask_case.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case] = case_code
        # apply no constrains
        # print("case 20: ", n_in_case)

    # -------------------- CASE 21 --------------------
    # have parent AND parent no constaint
    case_code = 21
    mask_case = np.logical_and.reduce((situation_code_recode == 2, ~parent_is_constraining))
    n_in_case = mask_case.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case] = case_code
        # apply no constrains
        # print("case 21: ", n_in_case)

    # -------------------- CASE 3 --------------------
    # n2 n3 n4 inv sqrt
    case_code = 3
    mask_case = np.logical_and.reduce(
        (situation_code_recode == 0, parent_is_constraining, parent_dimension_analysis_case == 3))
    n_in_case = mask_case.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case] = case_code
        tiled_power = np.tile(parent_power, reps=(7, 1)).transpose()
        apply_constraints(mask_case, True, parent_phy_units / tiled_power)
        # print("case 3: ", n_in_case)

    # -------------------- CASE 4 --------------------
    # abs neg max min
    case_code = 4
    mask_case = np.logical_and.reduce(
        (situation_code_recode == 0, parent_is_constraining, parent_dimension_analysis_case == 4))
    n_in_case = mask_case.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case] = case_code
        apply_constraints(mask_case, True, parent_phy_units)
        # print("case 4: ", n_in_case)

    # -------------------- CASE 5 --------------------
    # dimensionless op, i.e. sin cos tan exp, they need dimensionless number in function
    case_code = 5
    mask_case = np.logical_and.reduce(
        (situation_code_recode == 0, mask_have_parent, parent_dimension_analysis_case == 5))
    n_in_case = mask_case.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case] = case_code
        apply_const_constraints(mask_case, True, 0)
        # print("case 5: ", n_in_case)

    # -------------------- CASE 22 -------------------- */
    # parent constrain AND have sibling AND left sibling not empty AND left sibling constraint
    case_code = 22
    mask_case = np.logical_and.reduce(
        (situation_code_recode == 2, parent_is_constraining))  # for mul add or div sub at head, can be passed
    # mul mul mul s v s0 (need parent_is_constraining, when input s0 step, we don't need constraint, because parent not constraint)

    mask_case_todo_assign_units_bottom_up = np.logical_and.reduce(
        (mask_case, sibling_not_empty, sibling_is_constraining == False))

    if sum(mask_case_todo_assign_units_bottom_up) > 0:
        # Determining subtrees on which to perform bottom up dimensional analysis's start and ends
        # Start of subtree is the sibling of the current token
        coords_start = np.stack((batch_pos, sibling_index), axis=0)[:, mask_case_todo_assign_units_bottom_up]
        # End of subtree is the token just before the current token (pos - 1)
        coords_end = np.stack((batch_pos, pos - 1), axis=0)[:, mask_case_todo_assign_units_bottom_up]
        # Perform dimensional analysis
        assign_units_bottom_up(programs=programs, coords_start=coords_start, coords_end=coords_end)
        parent_is_constraining, parent_phy_units, parent_power, parent_dimension_analysis_case = get_parent_info()
        sibling_is_constraining, sibling_phy_units, sibling_dimension_analysis_case = get_sibling_info()
    n_in_case = mask_case.sum()
    # if n_in_case > 0:
    # print("case 22: ", n_in_case)

    # -------------------- CASE 220 -------------------- *
    case_code = 220
    mask_case_mul = np.logical_and.reduce((mask_case, parent_dimension_analysis_case == 20))
    n_in_case = mask_case_mul.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case_mul] = case_code
        apply_constraints(mask_case_mul, True, (parent_phy_units - sibling_phy_units))
        # print("case 220: ", n_in_case)

    # -------------------- CASE 221 -------------------- /
    case_code = 221
    mask_case_div = np.logical_and.reduce((mask_case, parent_dimension_analysis_case == 21))
    n_in_case = mask_case_div.sum()
    if n_in_case > 0:
        situation_code_recode[mask_case_mul] = case_code
        apply_constraints(mask_case_div, True, (sibling_phy_units - parent_phy_units))
        # print("case 221: ", n_in_case)

    # -------------------- ASSIGNMENT --------------------
    mask_need_to_be_assigned = programs.library.coords_to_mask(coords)
    # mask : is token in VectPrograms already constraining ?
    mask_tokens_already_constraining = programs.library.is_constraining_phy_units  # (batch_size, max_time_step,)
    # ignore the token that have been constraint
    mask_assign = mask_need_to_be_assigned & (mask_tokens_already_constraining == False)  # haven't already constraint
    curr_token_already_constraining = programs.library.is_constraining_phy_units[
        tuple(coords)]  # haven't already constraint
    programs.library.is_constraining_phy_units[mask_assign] = is_constraining[
        curr_token_already_constraining == False]  # (n_tokens_already_constraining,)
    programs.library.phy_units[mask_assign] = phy_units[
        curr_token_already_constraining == False]  # (n_tokens_already_constraining, UNITS_VECTOR_SIZE)
    programs.library.units_analysis_cases[tuple(coords)] = situation_code_recode
    return situation_code_recode


def assign_units_bottom_up(programs, coords_start, coords_end):
    """
        Performs a bottom up (in the tree representation of programs) dimensional analysis and assigns units along the way
        for multiple subtrees.
        Parameters
        ----------
        programs : program.VectPrograms
            Programs on which bottom up units assignment should be performed.
        coords_start : numpy.array of shape (2, ?) of int
            Coords of starts of subtrees, 0th array in batch dim and 1th array in time dim.
        coords_end : numpy.array of shape (2, ?) of int
            Coords of ends of subtrees, 0th array in batch dim and 1th array in time dim.
        Returns
        -------
        """
    # Assertions
    assert coords_start.shape[1] == coords_end.shape[1], "%i subtrees starts but %i subtrees ends were given." % (
        coords_start.shape[1], coords_end.shape[1])
    assert np.array_equal(coords_start[0], coords_end[0]), "Start and end of subtrees should be located on the same " \
                                                           "program."
    # Arguments
    n_subtrees = coords_start.shape[1]
    batch_pos = coords_start[0]
    start_pos = coords_start[1]
    end_pos = coords_end[1]

    # Iterating through subtrees
    # vectorize bottom-up units assignment process
    # t0 = time.perf_counter()
    for k in range(n_subtrees):
        # Starting at the end of subtree and parsing in reverse polish notation
        start = end_pos[k]  # replace start and end
        # End of parsing is start of subtree
        end = start_pos[k]
        prog_i = batch_pos[k]
        # Error messages
        error_msg_unknown_dim = "Unknown physical units token encountered in bottom up units assignment " \
                                "process."
        error_msg_dimensionless_child = "Non-dimensionless token encountered as child of dimensionless op (eg cos, " \
                                        "exp, log etc) in bottom up units assignment process."
        error_msg_dimensionless_token = "Dimensionless token having non-zero physical units encountered in bottom " \
                                        "up units assignment process."
        error_msg_additive_discrepancy = "Two children of binary_additive_op (eg: addition, subtraction) having " \
                                         "different physical units encountered in bottom up units assignment process."
        error_msg_incomplete_tree = "Regular bottom up dimensional analysis can not be performed on incomplete " \
                                    "tree (containing terminal tokens with unknown physical units: eg. dummies)"

        # Utils function to assign units
        def assign_units(pos, phy_units, is_constraining):
            programs.library.phy_units[prog_i, pos] = phy_units
            programs.library.is_constraining_phy_units[prog_i, pos] = is_constraining
            return None

        # Parser
        def parser(i):
            # Stop condition (getting outside of subtree)
            if i == end - 1:
                return True
            # Current parsing
            # parent info
            idx = programs.library.tokens_idx[prog_i, i]
            arity = programs.library.all_token_info_table.arity_table[idx]
            dimension_analysis_case = programs.library.all_token_info_table.dimension_analysis_case_table[idx]
            phy_units = programs.library.phy_units[prog_i, i]
            is_constraining = programs.library.is_constraining_phy_units[prog_i, i]
            # Check that subtree is complete (no dummies should be encountered during parsing)
            assert idx != programs.library.all_token_info_table.invalid_id, error_msg_incomplete_tree
            # Arity = 0 ---
            if arity == 0:
                pass
            # Arity = 1 ---
            elif arity == 1:
                # Position of the lonely child of the token (arity = 1)
                child_index = programs.library.children_index[prog_i, i][0]
                # child_idx = programs.library.tokens_idx[prog_i, child_index]
                child_phy_units = programs.library.phy_units[prog_i, child_index]  # should not be _table
                child_is_constraining = programs.library.is_constraining_phy_units[
                    prog_i, child_index]  # should not be _table, because they have update from bottom
                # Making sure that the child of unary tokens are not free
                assert child_is_constraining == True, error_msg_unknown_dim
                # If token is an unary power op -> apply power to units
                if dimension_analysis_case == 3:
                    n_power = programs.library.all_token_info_table.power_table[idx]
                    assign_units(pos=i, phy_units=n_power * child_phy_units, is_constraining=True)
                # Elif token is an unary additive op -> copy-paste units from child
                elif dimension_analysis_case == 4:  # same as child
                    assign_units(pos=i, phy_units=child_phy_units, is_constraining=True)
                # Elif token is an unary dimensionless op -> nothing to do but making sure that child token is
                # dimensionless (as it should be) just in case and that current token is dimensionless
                elif dimension_analysis_case == 5:  # need to be dimensionless
                    assert np.array_equal(child_phy_units, np.zeros(7)) and \
                           child_is_constraining == True, error_msg_dimensionless_child
                    assert np.array_equal(phy_units, np.zeros(7)) and \
                           is_constraining == True, error_msg_dimensionless_token
            # Arity = 2 ---
            elif arity == 2:
                # Children positions
                child0_index = programs.library.children_index[prog_i, i][0]
                child1_index = programs.library.children_index[prog_i, i][1]
                # Children idx
                # child0_idx = programs.library.tokens_idx[prog_i, child0_index]
                # child1_idx = programs.library.tokens_idx[prog_i, child1_index]
                # Child 0 units
                child0_phy_units = programs.library.phy_units[prog_i, child0_index]
                child0_is_constraining = programs.library.is_constraining_phy_units[prog_i, child0_index]
                # Child 1 units
                child1_phy_units = programs.library.phy_units[prog_i, child1_index]
                child1_is_constraining = programs.library.is_constraining_phy_units[prog_i, child1_index]
                # Assertion: making sure that children of binary tokens are not free
                assert (child0_is_constraining == True) and (
                        child1_is_constraining == True), error_msg_unknown_dim  # two need to be
                # If token is an additive token -> units are those of any children (as they should be the same
                # among them) but making sure that children of additive binary tokens have the same units for safety.
                if dimension_analysis_case == 1:
                    assert np.array_equal(child1_phy_units, child0_phy_units), error_msg_additive_discrepancy
                    assign_units(pos=i, phy_units=child0_phy_units, is_constraining=True)  # two children are added
                # Elif token is a multiplicative token
                elif dimension_analysis_case == 20 or dimension_analysis_case == 21:
                    # token = child0 * child1 => units(token) = child0_phy_units + child1_phy_units
                    if dimension_analysis_case == 20:
                        assign_units(pos=i, phy_units=child0_phy_units + child1_phy_units, is_constraining=True)
                    # token = child0 / child1 => units(token) = child0_phy_units - child1_phy_units
                    elif dimension_analysis_case == 21:
                        assign_units(pos=i, phy_units=child0_phy_units - child1_phy_units, is_constraining=True)
            # print("------")
            # print("Parsing i = ",i)
            # print("name      = ", programs.lib_names[programs.tokens.idx[prog_i, i]])
            # print("units     = ", programs.tokens.phy_units[prog_i, i])
            # Parsing previous token afterwards
            parser(i - 1)

        parser(start)
    # t1 = time.perf_counter()
    # print("assign_units_bottom_up %f ms"%((t1-t0)*1e3))
    return None
