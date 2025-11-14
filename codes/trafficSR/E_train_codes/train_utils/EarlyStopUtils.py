import numpy as np
def early_stop(result_record, stop_reward, stop_after_n_epochs, R_similarity, logging, visualiser):
    bool_early_stop = False
    """early stop if stop_reward is reached"""
    # early_stop_reward_eps = 2 * np.finfo(np.float32).eps
    # If above stop_reward (+/- eps) stop after [stop_after_n_epochs] epochs.
    if len(result_record.trainResult["best_R_history"])>0 and result_record.trainResult["best_R_history"][-1]>= stop_reward:
        if stop_after_n_epochs == 0:
            print("Early stop!")
            bool_early_stop = True
        stop_after_n_epochs -= 1

    """early stop if similarity: when have found IDM expression, break the loop"""
    if R_similarity.shape[1] > 0 and R_similarity[:, 0].max() >= 0.99:
        print("Find IDM Expression!")
        logging.info(f"Find Old Expression!")
        # breakout(visualiser, result_record.trainResult)
        # return result_record.trainResult["candidates_history"], np.array(
        #     result_record.trainResult["rewards_history"])

    """early stop if rmse==1"""
    # if np.any(R_SUB[:, 2] >= 0.995):
    #     print("Find Expression!")
    #     breakout(visualiser, result_record.trainResult)
    #     return result_record.trainResult["candidates_history"], np.array(
    #         result_record.trainResult["rewards_history"])
    
    return stop_after_n_epochs,bool_early_stop