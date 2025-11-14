import os
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KernelDensity
from IPython.display import display, clear_output

def print_c(data, color=31):
    """
    颜色样式打印输出功能
    :param data: 打印内容
    :param color: 指定颜色, 默认为红色(31)绿色(32)黄色(33)蓝色(34)
    :return:
    """
    if isinstance(color, int):
        color = str(color)
    print(f"\033[1;{color}m{data}\033[0m")


simplify_True_length_limit = 30
# Fig params
try:
    # None
    plt.rc('text', usetex=False)  # if True, latex lost lead to error
    plt.rc('font', family='serif')
    plt.rc('font', size=16)
except:
    warnings.warn("Latex display not available.")

# Faster than searching for best loc
LEGEND_LOC = 'upper left'  # "best"


class Visualiser:
    def __init__(self,
                 epoch_refresh_rate_plot=1,
                 epoch_refresh_rate_prints=1,
                 save_path=None,
                 do_prints=True,
                 do_plot=True,
                 do_save=True,
                 draw_all_progs_fit=True
                 ):
        self.epoch_refresh_rate_plot = epoch_refresh_rate_plot
        self.epoch_refresh_rate_prints = epoch_refresh_rate_prints
        # self.figsize = (40, 10)
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_path is not None:
            self.save_path_log = save_path + "data.csv"  # save_path with extension replaced by '_data.csv'
            self.save_path_pareto = save_path + "pareto.csv"  # save_path with extension replaced by '_pareto.csv'
            self.save_path_pareto_fig = save_path + "pareto.pdf"  # save_path with extension replaced by '_pareto.pdf'
            self.save_path_demo_pic = save_path + "demo.png"  # save_path with extension replaced by '_demo.png'
            self.save_path_score_density_pic = save_path + "score_density"  # save_path with extension replaced by '_demo.png'
        self.do_plot = do_plot
        self.do_save = do_save
        self.do_prints = do_prints
        self.draw_all_progs_fit = draw_all_progs_fit

    def initialize(self):
        self.epoch_history = []
        # self.fig = plt.figure(figsize=self.figsize)
        self.fig = plt.figure(figsize=(12, 9))
        gs = gridspec.GridSpec(1, 1)
        self.ax0 = self.fig.add_subplot(gs[0, 0])
        div = make_axes_locatable(self.ax0)
        self.cax = div.append_axes("right", size="4%", pad=0.4)
        # self.ax1 = self.fig.add_subplot(gs[0, 1])
        # self.ax2 = self.fig.add_subplot(gs[0, 2])
        self.t0 = time.perf_counter()

    def make_prints(self, trainResult):
        t1 = self.t0
        t2 = time.perf_counter()
        self.t0 = t2

        print_c("-> Time %.2f s" % (t2 - t1), color=32)
        # Best of epoch
        if trainResult["rewards_history"][-1] > 0:
            print_c("-> Best reward of epoch=%f" % (trainResult["rewards_history"][-1]), color=32)
            print("Best expression of epoch : \n%s" % (
                trainResult["candidates_history"][-1].get_print_expression(
                    do_simplify=True if len(
                        trainResult["candidates_history"][-1].tokens) < simplify_True_length_limit else False)))
            program_free_const_str, program_semi_free_const_str, program_fixed_const_str = \
                trainResult["candidates_history"][-1].get_print_const()
            # program_free_const_str = trainResult["free_const_history"][-1][0]
            if program_free_const_str:
                print("free_const of program:", set(program_free_const_str))
            if program_semi_free_const_str:
                print("semi_free_const of program:", set(program_semi_free_const_str))
            if program_fixed_const_str:
                print("fixed_const of program:", set(program_fixed_const_str))

        # Overall best
        if trainResult["best_R_history"][-1] > 0:
            print_c("-> Overall best reward=%f" % (trainResult["best_R_history"][-1]), color=32)
            # print("-> Overall best expression : \n%s" % (
            #     trainResult["best_program"].get_print_expression(do_simplify=False)))
            print("Overall best expression(simplified) : \n%s" % (
                trainResult["best_program"].get_print_expression(
                    do_simplify=True if len(
                        trainResult[
                            "best_program"].tokens) < simplify_True_length_limit else False)))  # solve inv 1*1/x presentation problem
            program_free_const_str, program_semi_free_const_str, program_fixed_const_str = \
                trainResult["best_program"].get_print_const()
            if program_free_const_str: # 这一行有问题，跟之前
                print("free_const of program:", set(program_free_const_str))
            if program_semi_free_const_str:
                print("semi_free_const of program:", set(program_semi_free_const_str))
            if program_fixed_const_str:
                print("fixed_const of program:", set(program_fixed_const_str))
            # print("  -> Simplified expression : \n%s"%(run_logger.best_prog.get_infix_pretty(do_simplify=True , )))
        print("\n")

    def update_plot(self, trainResult):

        # -------- Reward distrbution vs epoch --------
        curr_ax = self.ax0
        cmap = plt.get_cmap("viridis")
        fading_plot_nepochs = self.epoch_history[-1]
        fading_plot_ncurves = 25
        fading_plot_max_alpha = 1.
        fading_plot_bins = 100
        fading_plot_kde_bandwidth = 0.05
        curr_ax.clear()
        self.cax.clear()
        # Plotting last "fading_plot_nepochs" epoch on "fading_plot_ncurves" curves
        plot_epochs = []
        for i in range(fading_plot_ncurves + 1):
            frac = (i / fading_plot_ncurves) * 0.6 + 0.4
            plot_epoch = int(self.epoch_history[-1] - frac * fading_plot_nepochs)
            plot_epochs.append(plot_epoch)
            prog = 1 - frac
            alpha = fading_plot_max_alpha * (prog)
            # Histogram
            bins_dens = np.linspace(0., 1, fading_plot_bins)
            kde = KernelDensity(kernel="gaussian", bandwidth=fading_plot_kde_bandwidth
                                ).fit(np.array(trainResult["R_train_history"])[plot_epoch][:, np.newaxis])
            dens = 10 ** kde.score_samples(bins_dens[:, np.newaxis])
            # Plot
            curr_ax.plot(bins_dens, dens, alpha=alpha, linewidth=0.8, c=cmap(prog))
        # Colorbar
        normcmap = plt.matplotlib.colors.Normalize(vmin=plot_epochs[0], vmax=plot_epochs[-1])
        cbar = self.fig.colorbar(plt.cm.ScalarMappable(norm=normcmap, cmap=cmap), cax=self.cax, pad=0.005)
        title_font = 16
        label_font = 18
        axis_font = 14
        cbar.set_label('Epoch', rotation=90, labelpad=30, fontsize=label_font)
        curr_ax.set_xlim([0, 0.8])
        curr_ax.set_xlabel("Weighted Score", fontsize=label_font)
        curr_ax.set_ylabel("Density", fontsize=label_font)
        curr_ax.grid(True, linewidth=0.1, linestyle='--')
        plt.xticks(fontsize=axis_font)
        plt.yticks(fontsize=axis_font)
        # plt.title("Density distribution of weighted score of candidate models in each epoch", fontsize=title_font, )

        # -------- Reward vs epoch --------
        # curr_ax = self.ax1
        # curr_ax.clear()
        # curr_ax.plot(self.epoch_history, trainResult["mean_R_history"], 'b', linestyle='solid', alpha=0.6,
        #              label="Mean reward of epoch")
        # curr_ax.plot(self.epoch_history, trainResult["mean_R_train_history"], 'r', linestyle='solid', alpha=0.6,
        #              label="Mean train reward of epoch")
        # curr_ax.plot(self.epoch_history, trainResult["rewards_history"], color='orange', linestyle='solid', alpha=0.6,
        #              label="Best reward of epoch")
        # curr_ax.plot(self.epoch_history, trainResult["best_R_history"], 'k', linestyle='solid', alpha=1.0,
        #              label="Overall best reward until now")
        #
        # curr_ax.set_ylabel("Reward")
        # curr_ax.set_xlabel("Epochs")
        # curr_ax.legend(loc=LEGEND_LOC)

        # -------- Loss --------
        # curr_ax = self.ax2
        # curr_ax.clear()
        # curr_ax.plot(self.epoch_history, trainResult["loss_history"], 'grey', label="loss")
        # curr_ax.set_ylabel("Loss")
        # curr_ax.set_xlabel("Epochs")
        # curr_ax.legend(loc=LEGEND_LOC)

        # -------- Display --------
        # plt.draw()
        # plt.show()
        # plt.pause(1)
        # plt.close()
        # following only work out for IPython or Jupyter notebook, so we use plt.show
        # plt.draw()
        # plt.pause(1)
        # display(self.fig)
        # clear_output(wait=True) # until next plot, then clear plot

    def show_visualization(self):
        display(self.fig)
        clear_output(wait=True)  # until next plot, then clear plot

    def save_visualization(self):
        self.fig.savefig(self.save_path_score_density_pic + ".png", bbox_inches='tight')
        self.fig.savefig(self.save_path_score_density_pic + ".eps", bbox_inches='tight')
        self.fig.savefig(self.save_path_score_density_pic + ".pdf", bbox_inches='tight')

    def visualise(self, epoch, trainResult):
        if epoch == 0:
            self.initialize()
        # update history
        self.epoch_history.append(epoch)
        # Prints Function
        if epoch % self.epoch_refresh_rate_prints == 0:
            try:
                if self.do_prints:
                    self.make_prints(trainResult)
            except:
                print("Unable to print function.")
        # Plot reward and loss
        if epoch % self.epoch_refresh_rate_plot == 0:
            try:
                self.update_plot(trainResult)
                if self.do_save:
                    self.save_visualization()
            except:
                print("Unable to plot.")
