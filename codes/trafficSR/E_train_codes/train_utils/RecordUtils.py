from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def log_print(all_str,logging,need_print=True):
    for i in all_str:
        logging.info(i)
        if need_print:
            print(i)

def envInitLogging(
    logging, logPath, formatted_time, data_source, evolution, env, train_args
):
    """initial log info"""
    logging.basicConfig(
        filename=logPath + f"{formatted_time}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    '''print log info'''    
    all_str=[
        f"Start time: {formatted_time}",
        f"Data source: {data_source}",
        f"Evolution: {evolution}",
        f"All tokens: {env.all_tokens}",
        f"Free const tokens: {train_args['token_args']['free_const_tokens']}",
        f"Initial value of free const tokens: {train_args['token_args']['free_const_initial_values']}",
        f"Semi-free const tokens: {train_args['token_args']['semi_free_const_tokens']}",
        f"Initial value of semi-free const values: {train_args['token_args']['semi_free_const_initial_values']}",
        f"Fixed const tokens: {train_args['token_args']['fixed_const_tokens']}",
        f"Fixed const values: {train_args['token_args']['fixed_const_values']}",
        f"batch_size: {train_args['env_args']['batch_size']}",
        f"max_time_step: {train_args['env_args']['max_time_step']}",
        f"risk factor: {train_args['env_args']['risk_factor']}",
        f"parrallel_mode: {train_args['env_args']['parallel_mode']}",
        f"n_cpus: {train_args['env_args']['n_cpus']}",
        f"reward_weight: {train_args['env_args']['reward_weight']}",
        "\n"
    ]
    log_print(all_str,logging,need_print=False)
    print("Start time:", formatted_time)
    print("Data source:", data_source)


def epoch_log_info(
    logging,
    evolution,
    epoch,
    R,
    R_SUB,
    best_prog_id,
    envs,
    loss_val,
    R_train,
    result_record,
    buffer_from_exploration_length,
    r2=0,
    overall_best_r2=0,
):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d %H-%M-%S")
    xrmse_length = int((R_SUB.shape[1] - 3) / 2)
    all_str = [
        f"Evolution-Epoch and Epoch time now: {evolution, epoch, formatted_time}",
        f"-> Epoch best reward, r2: {R.max(), r2}",
        f"Epoch best program: {envs.main_env.library.prefix_str[best_prog_id]}",
        f"Sub-Reward(complexity similarity rmse xrmse vrmse) of epoch_best: {R_SUB[best_prog_id, :]}",
        f"Mean sub-reward(xrmse vrmse) of epoch_best: {np.mean(R_SUB[best_prog_id, 3:3 + xrmse_length])}, {np.mean(R_SUB[best_prog_id, 3 + xrmse_length:])}",
        "\n",
        f"-> R_train mean/ R_train min/ R mean: {R_train.mean(),  R_train.min(), R.mean(),}",
        f"n_physical/ n_elites/ n_from_buffer: {sum(envs.main_env.library.is_physical), sum(R_train != 0), buffer_from_exploration_length}",
        f"Mean Sub_Reward of elites: {result_record.trainResult['mean_R_train_sub_history'][-1][:3]}",
        f"Train loss_gp: {loss_val}",
        "\n",
        f"-> Overall best reward, r2 : {result_record.trainResult['best_R_history'][-1], overall_best_r2}",
        f"Best program of overall : {result_record.trainResult['best_program']}",
        f"Sub-Reward(complexity similarity rmse xrmse vrmse) of overall_best: {result_record.trainResult['best_sub_R_history'][-1][:]}",
        f"Mean sub-reward(xrmse vrmse) of overall_best: {np.mean(result_record.trainResult['best_sub_R_history'][-1][3:3 + xrmse_length])}, {np.mean(result_record.trainResult['best_sub_R_history'][-1][3 + xrmse_length:])}",
        f"Acceleration_loss/Velocity_loss/PositionX_loss of overall_best: {result_record.best_a_loss, result_record.best_v_loss, result_record.best_x_loss}",
        "\n"
    ]
    log_print(all_str, logging, need_print=True)
    print("\n")


def evolution_log_info(logging, trainResult, best_free_const_values, best_a_loss):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d %H-%M-%S")
    all_str = [
        f"Evolution end time: {formatted_time}",
        f"This evolution is done, best program is: {trainResult['best_program']}",
        f"Free const values of the best program: {best_free_const_values}",
        f"Loss of the best program: {best_a_loss}",
        "\n"
    ]
    log_print(all_str, logging, need_print=True)
    print("\n")


def write_evolution_different_program_md(
    trainResult, agent, md_folder="", evolution=0, save_number=20
):
    writePath = md_folder + "different_expressions_evolution" + f"{evolution}.md"
    agent.write_evolution_expressions(
        trainResult=trainResult, writePath=writePath, save_number=save_number
    )


def plot_pareto_front(
    run_logger,
    do_simplify=True,
    show_superparent_at_beginning=True,
    eq_text_size=12,
    delta_xlim=[0, 5],
    delta_ylim=[0, 15],
    frac_delta_equ=[0.03, 0.03],
    figsize=(20, 10),
    ax=None,
    fig=None,
    color="r",
):
    (
        pareto_front_complexities,
        pareto_front_programs,
        pareto_front_r,
        pareto_front_rmse,
    ) = run_logger.get_pareto_front()

    pareto_front_rmse = pareto_front_rmse
    # Fig params
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # # enables new_dummy_symbol = "\square"
    # plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{xcolor}')
    # plt.rc('font', size=32)

    # Fig
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    #     matplotlib.pyplot.close()
    # RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
    #   fig, ax = plt.subplots(1, 1, figsize=figsize)

    if len(pareto_front_complexities) == 0:
        return fig, ax

    ax.plot(pareto_front_complexities, pareto_front_rmse, f"{color}-")
    ax.plot(pareto_front_complexities, pareto_front_rmse, f"{color}o")

    # Limits
    xmin = 0 + delta_xlim[0]
    xmax = run_logger.batch.max_time_step + delta_xlim[1]
    ymin = 0 + delta_ylim[0]
    ymax = 10 + delta_ylim[1]
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    # Axes labels
    ax.set_xlabel("Sampling Length")
    ax.set_ylabel("RMSE")

    for i_prog in range(len(pareto_front_programs)):
        prog = pareto_front_programs[i_prog]

        text_pos = [
            pareto_front_complexities[i_prog] + frac_delta_equ[0] * (xmax - xmin),
            pareto_front_rmse[i_prog] + frac_delta_equ[1] * (ymax - ymin),
        ]
        # Getting latex expr
        latex_str = prog.get_infix_latex()
        # Adding "superparent =" before program to make it pretty
        if show_superparent_at_beginning:
            # latex_str = prog.library.superparent.name + ' =' + latex_str
            latex_str = latex_str

        ax.text(text_pos[0], text_pos[1], f"${latex_str}$", size=eq_text_size)

    return fig, ax
