import torch
import numpy as np
from torch.nn import functional as F
from codes.trafficSR.A_sampling.SAC_utils.SAC_value import ValueNetwork
from codes.trafficSR.A_sampling.SAC_utils.ReplayBuffer import ReplayBuffer
from codes.trafficSR.A_sampling.policy_network.policy_network import PolicyNetwork

# SAC最大化预期回报的同时最大化策略的熵来鼓励探索, 使得策略更加鲁棒和稳定


class SAC:
    def __init__(
        self,
        policy_network: PolicyNetwork,
        train_args,
        action_space_dim,
        device,
        dtype,
        envs,
        prior_args,
    ):
        # 属性分配
        self.sac_args = train_args["sac_args"]
        self.actor_lr = self.sac_args["actor_lr"]
        self.critic_lr = self.sac_args["critic_lr"]
        self.alpha_lr = self.sac_args["alpha_lr"]

        self.action_space_dim = action_space_dim
        # self.target_entropy = self.sac_args["target_entropy"]
        self.target_entropy = (
            -np.log(1.0 / self.action_space_dim) * 0.98
        )  # 对于离散的应该这样求，连续的是另一种：-action_dim（动作空间维度）
        self.gamma = self.sac_args["gamma"]
        self.tau = self.sac_args["tau"]
        self.n_warmup_batches = self.sac_args["n_warmup_batches"]
        self.sample_batch_size = self.sac_args["sample_batch_size"]
        self.min_samples = self.n_warmup_batches * self.sample_batch_size
        self.device = device

        # 环境 for prior
        self.envs = envs

        # 实例化策略网络
        self.actor = policy_network

        # SAC使用了两个Q网络（Critic）
        # 实例化第一个价值网络--预测
        self.critic_1 = ValueNetwork(
            train_args,
            action_space_dim,
            device,
            dtype,
        )
        # 实例化第二个价值网络--预测
        self.critic_2 = ValueNetwork(
            train_args,
            action_space_dim,
            device,
            dtype,
        )

        # 以及对应的目标网络:目标网络，它们是主网络的一个移动平均版本
        # 实例化价值网络1--目标
        self.target_critic_1 = ValueNetwork(
            train_args,
            action_space_dim,
            device,
            dtype,
        )
        # 实例化价值网络2--目标
        self.target_critic_2 = ValueNetwork(
            train_args,
            action_space_dim,
            device,
            dtype,
        )
        # 预测和目标的价值网络的参数初始化一样
        self.target_critic_1.model.load_state_dict(self.critic_1.model.state_dict())
        self.target_critic_2.model.load_state_dict(self.critic_2.model.state_dict())

        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.model.parameters(), lr=self.actor_lr
        )
        # 目标网络的优化器
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.model.parameters(), lr=self.critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.model.parameters(), lr=self.critic_lr
        )

        # 同时最大化预期回报和策略的熵,通过调整一个可学习的温度参数alpha来实现.用来动态调整策略探索程度
        # 初始化可训练参数alpha
        self.log_alpha = torch.tensor(
            0.0, dtype=torch.float, requires_grad=True
        )  # np.log(0.01)
        # 定义alpha的优化器
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
    
    # 更新replay_buffer的绝对TD误差，用于优先经验回放
    def update_abs_td_error(self, replay_buffer:ReplayBuffer):
        transitions = [replay_buffer.buffer[idx] for idx in range(replay_buffer.len())] # 采样全部数据
        transition_dict=replay_buffer.transform_transition2dict(transitions) # 样本对转换为字典
        (
            ob_overall,
            ob_partial,
            actions,
            rewards,
            next_ob_overall,
            next_ob_partial,
            dones,
            states_overall,
            states_partial,
            next_states_overall,
            next_states_partial,
            prior_UCBs,
            next_prior_UCBs,
            _,
        ) = self.decode_transition_dict(transition_dict) # 解码样本字典，并形成torch变量
        
        # 计算td_target(根据策略网络和目标网络)
        td_target = self.calc_target(
            rewards,
            next_ob_partial,
            next_ob_overall,
            next_states_partial,
            next_states_overall,
            dones,
            next_prior_UCBs,
        )
        
        # 当前网络预测的state_value
        critic_1_qvalues = self.critic_1.model(ob_partial, ob_overall).gather(
            dim=1, index=actions
        )
        critic_2_qvalues = self.critic_2.model(ob_partial, ob_overall).gather(
            dim=1, index=actions
        )
        # 计算TD误差
        abs_td_errors=0.5*(torch.abs(critic_1_qvalues-td_target)+torch.abs(critic_2_qvalues-td_target))
        replay_buffer.update_abs_td_error(abs_td_errors.detach().numpy()) # 更新TD误差
        
    # 从字典中,解码经验池样本所包含的数据集
    def decode_transition_dict(self, transition_dict):
        ob_overall = torch.tensor(
            np.vstack(transition_dict["obsers_overall"]), dtype=torch.float
        ).to(self.device)  # [b,n_states] 60
        ob_partial = torch.tensor(
            np.vstack(transition_dict["obsers_partial"]), dtype=torch.float
        ).to(
            self.device
        )  # [b,n_states] 36
        actions = (
            torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        )  # [b,1]
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )  # [b,1]
        next_ob_overall = torch.tensor(
            np.vstack(transition_dict["next_obsers_overall"]), dtype=torch.float
        ).to(
            self.device
        )  # [b,n_states] 60
        next_ob_partial = torch.tensor(
            np.vstack(transition_dict["next_obsers_partial"]), dtype=torch.float
        ).to(
            self.device
        )  # [b,n_states] 36
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )  # [b,1]
        states_overall = (
            torch.stack(transition_dict["states_overall"], dim=0)
            .permute(1, 2, 0, 3)
            .to(self.device)
        )  # [n_layers,2,b,hidden_size=64]
        states_partial = (
            torch.stack(transition_dict["states_partial"], dim=0)
            .permute(1, 2, 0, 3)
            .to(self.device)
        )  # [n_layers,2,b,hidden_size=64]
        next_states_overall = (
            torch.stack(transition_dict["next_states_overall"], dim=0)
            .permute(1, 2, 0, 3)
            .to(self.device)
        )  # [n_layers,2,b,hidden_size=64]
        next_states_partial = (
            torch.stack(transition_dict["next_states_partial"], dim=0)
            .permute(1, 2, 0, 3)
            .to(self.device)
        )  # [n_layers,2,b,hidden_size=64]
        prior_UCBs = torch.stack(transition_dict["prior_UCBs"], dim=0).to(
            self.device
        )  # [b,n_actions]
        next_prior_UCBs = torch.stack(transition_dict["next_prior_UCBs"], dim=0).to(
            self.device
        )  # [b,n_actions]
        abs_td_errors = torch.tensor(transition_dict["abs_td_errors"], dtype=torch.float).to(self.device) # [b,1]
        return (
            ob_overall,
            ob_partial,
            actions,
            rewards,
            next_ob_overall,
            next_ob_partial,
            dones,
            states_overall,
            states_partial,
            next_states_overall,
            next_states_partial,
            prior_UCBs,
            next_prior_UCBs,
            abs_td_errors,
        )
    
    def probs_from_logits(self,logits):
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-3) # 防止log(0),需要加上一个很小的数,log(1e-3)=-6.9078
        return probs, log_probs
    
    # 计算目标，当前状态下的state_value:利用策略网络预测下一时刻的动作概率，价值网络预测下一时刻的动作价值（取两者中的最小值），再计算下一时刻的熵相关。从而综合起来，计算当前时刻的state_value=reward+(1-dones) * gamma*target_q_values。可以视为真值，是我们希望达到的。
    # 其实原写法计算没问题，但是比较难看懂
    def calc_target(
        self,
        rewards,
        
        next_ob_partial,
        next_ob_overall,
        next_states_partial,
        next_states_overall,
        dones,
        next_prior_UCBs,
    ):
        """核心是要获得策略网络预测的下一时刻动作概率 [b,n_actions]"""
        # 策略网络预测下一时刻的state_value  [b,n_states]-->[b,n_actions]
        next_logits, _, _ = self.actor.model(
            next_ob_partial,
            next_ob_overall,
            next_states_partial,
            next_states_overall,
        ) # model_logits
        next_logits = next_logits + next_prior_UCBs  # action_logits, not model_logits
        next_probs, next_log_probs = self.probs_from_logits(next_logits)

        """核心是要获得价值网络预测的下一时刻状态价值，取两者中的最小值 [b,1]"""
        # 这里用目标价值网络！！！，下一时刻的state_value [b,n_actions]
        q1_value = self.target_critic_1.model(next_ob_partial, next_ob_overall)
        q2_value = self.target_critic_2.model(next_ob_partial, next_ob_overall)
        # 取出最小的q值  [b, n_actions]
        min_q_values = torch.min(q1_value, q2_value)  # QSA_min

        """核心是要获得下一时刻的state_value [b,1]"""
        target_q_values = torch.sum(
            next_probs * (min_q_values - self.log_alpha.exp() * next_log_probs),
            dim=1,
            keepdim=True,
        ) # 这里需要self.log_alpha.exp()

        """时序差分，输出当前时刻的state_value  [b, 1]"""
        td_target = (dones) * rewards + (1 - dones) * self.gamma * target_q_values # 如果dones是1，那么下一时刻的state_value就直接是reward，否则还需要估计下一时刻的state_value
        # 感觉这个rewards不对，不应该整个序列一样rewards; 比如说没有结束的话，dones=0，则中间状态rewards视为0，只有最后一个状态的reward才有意义（纯靠贝尔曼进行传播了）。这种写法能保证中间状态也能被选取到
        return td_target

    # 更新价值网络，价值网络预测当前状态下的动作价值，再和当前真值td_target计算损失。从而向td_target靠拢
    def update_critics(self, ob_partial, ob_overall, actions, td_target):
        # 价值网络预测！！！，当前状态下经验池所使用动作的价值  [b, 1]
        critic_1_qvalues = self.critic_1.model(ob_partial, ob_overall).gather(
            dim=1, index=actions
        )
        critic_2_qvalues = self.critic_2.model(ob_partial, ob_overall).gather(
            dim=1, index=actions
        )
        # 均方差损失 预测-目标 only a single value
        critic_1_loss = F.mse_loss(critic_1_qvalues, td_target.detach()) # 需要td_target.detach(), 会在计算均方误差损失（MSE Loss）时使用了一个与计算图分离的（即没有梯度流经的）目标值。detach后，不会对产生目标值的Actor和Alpha的梯度计算产生影响，而是调整critic网络的参数.
        critic_2_loss = F.mse_loss(critic_2_qvalues, td_target.detach())
        # 梯度清0
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        # 梯度反传
        critic_1_loss.backward() # 不进行detach会报错（Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)
        critic_2_loss.backward()
        # 梯度更新
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        return critic_1_loss, critic_2_loss
    
    # 计算策略网络的熵，用于更新alpha
    def actor_prob_current(self,ob_partial,ob_overall,states_partial,states_overall,prior_UCBs):
        logits, _, _ = self.actor.model(
            ob_partial,
            ob_overall,
            states_partial,
            states_overall,
        )  # 预测当前时刻的动作概率
        logits = logits + prior_UCBs
        probs, log_probs = self.probs_from_logits(logits)
        # entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        return probs, log_probs
    
    def q_estimite_current(self,ob_partial,ob_overall,probs):
        # --------------------------------- #
        # 价值网络预测当前时刻的state_value——about each action  [b,n_actions]
        # --------------------------------- #
        q1_value = self.critic_1.model(
            ob_partial,
            ob_overall,
        )
        q2_value = self.critic_2.model(
            ob_partial,
            ob_overall,
        )
        # --------------------------------- #
        # 价值网络预测当前时刻的总state_value estimate [b,1]
        # --------------------------------- #
        qvalue = torch.sum(
            probs * torch.min(q1_value, q2_value), dim=1, keepdim=True
        )  # QS_min
        return qvalue
    
    def update_actor(self, log_probs, qvalue, probs):
        # probs * log_probs是策略网络的熵. probs已经是softmax后的概率，log_probs是对应的logsoftmax对数概率；# 之前的写法错了：因为应该是probs*inside_term
        # 写法2
        # inside_term=self.log_alpha.exp() * log_probs - qvalue
        # actor_loss=torch.mean(probs * inside_term) 
        # 写法1
        actor_loss=(self.log_alpha.exp() * probs * log_probs - qvalue).sum(dim=1).mean() # 这里需要self.log_alpha.exp()
        # 梯度更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss
    
    def update_alpha(self, log_probs):
        target_alpha = (log_probs + self.target_entropy).detach()
        alpha_loss = torch.mean(-self.log_alpha * target_alpha) # -self.log_alpha.exp() * target_alpha [b,n_actions]; 这里不要self.log_alpha.exp()
        # 梯度更新
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return alpha_loss

    # 软更新，每次训练，用Critic网络缓慢更新目标网络
    def soft_update(self, net, target_net):
        # 用于缓慢更新目标网络的参数，critic是两个预测网络，target_critic是两个目标网络
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1 - self.tau) + param.data * self.tau
            ) # 预测网络的参数赋给目标网络

    # 模型训练
    def update(self, transition_dict, bool_actor_update=True):
        # 提取数据集,prior_UCBs.requires_grad=False
        (
            ob_overall,
            ob_partial,
            actions,
            rewards,
            next_ob_overall,
            next_ob_partial,
            dones,
            states_overall,
            states_partial,
            next_states_overall,
            next_states_partial,
            prior_UCBs,
            next_prior_UCBs,
            abs_td_errors,
        ) = self.decode_transition_dict(transition_dict)

        # --------------------------------- #
        # 计算输出当前时刻的state_value [b, 1]
        # --------------------------------- #
        td_target = self.calc_target(
            rewards,
            next_ob_partial,
            next_ob_overall,
            next_states_partial,
            next_states_overall,
            dones, # 0代表未结束，1代表结束
            next_prior_UCBs,
        )

        # --------------------------------- #
        # 更新2个价值网络:价值网络预测当前状态下的动作价值，再和td_target计算损失
        # --------------------------------- #
        critic_1_loss, critic_2_loss = self.update_critics(
            ob_partial, ob_overall, actions, td_target
        )

        # --------------------------------- #
        # 计算策略网络损失，并进行更新
        # --------------------------------- #
        # 计算策略网络的动作概率 [b,n_actions]
        probs, log_probs = self.actor_prob_current(ob_partial,ob_overall,states_partial,states_overall,prior_UCBs)
        if bool_actor_update:
            # 价值网络预测当前时刻的总state_value estimate [b,1]
            qvalue = self.q_estimite_current(ob_partial,ob_overall,probs)
            # 计算策略网络loss
            actor_loss = self.update_actor(log_probs, qvalue, probs)

        # --------------------------------- #
        # 更新可训练遍历alpha
        # --------------------------------- #
        alpha_loss = self.update_alpha(log_probs)
        
        # --------------------------------- #
        # 软更新目标价值网络
        # --------------------------------- #
        self.soft_update(self.critic_1.model, self.target_critic_1.model)
        self.soft_update(self.critic_2.model, self.target_critic_2.model)

        # --------------------------------- #
        # print loss
        # --------------------------------- #
        actor_loss_numpy = actor_loss.detach().cpu().numpy() if bool_actor_update else 0
        print(f"loss critic_1/critic_2/actor/alpha: {critic_1_loss.detach().cpu().numpy(),critic_2_loss.detach().cpu().numpy(),actor_loss_numpy,alpha_loss.detach().cpu().numpy()}\n")
        return actor_loss_numpy
