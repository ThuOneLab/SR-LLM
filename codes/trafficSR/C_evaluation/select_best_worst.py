from codes.trafficSR.C_evaluation.SRscore import SRscore


def LLM_get_new_combinations(agent, train_args, token_args, evolution):
    new_combinations = {}
    # llm extract valuable combination:old method
    if train_args["bool_args"]["bool_extract_valuable"]:
        agent.extract_valuable_old(new_tokens_number=train_args["bool_args"]["new_tokens_number"],
                                   origin_token_dict=token_args,
                                   trainResultFile=f'different_expressions_evolution{evolution}.md',
                                   evolution=evolution)
        '''read in new_combinations info'''
        code_blocks = agent.extract_python_code_blocks(
            f"{agent.trainResultFolder}extract_valuable{evolution}.md")
        new_combinations = eval(code_blocks[-1])

    # One step increment of combinations: new method
    elif train_args["bool_args"]["bool_agent_single_step_increase"]:
        agent.single_step_increase(new_tokens_number=train_args["bool_args"]["new_tokens_number"],
                                   origin_token_dict=token_args,
                                   trainResultFile=f'different_expressions_evolution{evolution}.md',
                                   evolution=evolution)
        '''read in new_combinations info'''
        code_blocks = agent.extract_python_code_blocks(
            f"{agent.trainResultFolder}single_step_increase{evolution}.md")
        new_combinations = eval(code_blocks[-1])

    '''Remove duplicate combinations with existing combinations in the library'''
    if "new_combinations" in new_combinations:
        new_combinations["new_combinations"] = {key: value for key, value in
                                                new_combinations["new_combinations"].items() if
                                                value["infix_expression"] not in token_args[
                                                    "combination_infix_expression"]}
    return new_combinations

'''useless'''
def LLM_select_best_combinations(agent, new_env, different_expression, reward_of_expressions, evolution,
                                 have_got_semantic_score, k_selection_of_combinations):
    '''initialize score'''
    score = SRscore(env=new_env, different_expression=different_expression, agent=agent,
                    reward_of_expressions=reward_of_expressions)
    '''evaluate numerical score'''
    score.get_numerical_score()
    '''evaluate semantic score'''
    have_got_semantic_score = score.get_semantic_score(evolution=evolution,
                                                       have_get_semantic_score=have_got_semantic_score)
    '''evaluate integrated score'''
    score.get_integrated_score()
    '''Select the top K combinations and generate updated_combination'''
    best_combinations = score.select_best_combinations(top_k=k_selection_of_combinations)
    '''save best_combinations'''
    agent.save_prompt(prompt=agent.dict2block(best_combinations),
                      filename=f"{agent.trainResultFolder}updated_combinations{evolution}.md")
    return have_got_semantic_score


def LLM_select_best_worst_for_rag(agent, new_env, different_expression, reward_of_expressions, evolution,
                                  have_got_semantic_score, k_best_of_symbols, k_delete_of_combinations,
                                  k_best_of_expressions, combination_UCB=None, directly_built=None):
    '''initialize score'''
    if directly_built is None:
        directly_built = []
    score = SRscore(env=new_env, different_expression=different_expression, agent=agent,
                    reward_of_expressions=reward_of_expressions)
    '''evaluate numerical score'''
    score.get_numerical_score()
    '''evaluate UCB score'''
    score.get_UCB_score(combination_UCB)
    '''evaluate semantic score'''
    # have_got_semantic_score = score.get_semantic_score(evolution=evolution,
    #                                                    have_get_semantic_score=have_got_semantic_score)
    '''evaluate integrated score'''
    score.get_integrated_score()
    '''Select the top K combinations and worst K and generate updated_combination'''
    best_symbols, worst_combinations, best_expressions = score.select_best_worst_combinations_rag(
        top_k=k_best_of_symbols,
        worst_k=k_delete_of_combinations,
        k_best_of_expressions=k_best_of_expressions,
        directly_built=directly_built)
    '''save best_combinations'''
    agent.save_prompt(prompt=agent.dict2block(best_symbols) + agent.dict2block(best_expressions),
                      filename=f"{agent.trainResultFolder}best_symbols{evolution}.md")
    if worst_combinations:
        agent.save_prompt(prompt=agent.dict2block(worst_combinations),
                          filename=f"{agent.trainResultFolder}worst_combinations{evolution}.md")
    return best_symbols, worst_combinations, best_expressions, have_got_semantic_score
