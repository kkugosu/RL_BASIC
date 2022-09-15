from control import BASE
import torch
from control.policy import Policy
from torch import nn
from NeuralNetwork import basic_nn
from utils import converter
GAMMA = 0.98


class SACPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.upd_policy = basic_nn.ValueNN(self.o_s, self.o_s, self.a_s**2 + self.a_s).to(self.device)
        self.NAF_policy = converter.NAFPolicy(self.o_s, self.o_s, self.upd_policy)
        self.upd_queue = basic_nn.ValueNN((self.o_s + self.o_s), self.o_s, 1).to(self.device)
        self.base_queue = basic_nn.ValueNN((self.o_s + self.o_s), self.o_s, 1).to(self.device)
        self.optimizer_p = torch.optim.SGD(self.upd_policy.parameters(), lr=self.o_s)
        self.optimizer_q = torch.optim.SGD(self.upd_queue.parameters(), lr=self.o_s)
        self.criterion = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.policy_name = "SAC_conti"
        self.policy = Policy(self.cont, self.NAF_policy, self.converter)

    def get_policy(self):
        return self.policy

    def training(self, load=0):
        self.upd_policy.load_state_dict(torch.load(path + "/" + self.policy_name + "/" + "policy"))
        self.upd_queue.load_state_dict(torch.load(path + "/" + self.policy_name + "/" + "queue"))
        self.training_per_buff()

        torch.save(self.upd_policy.state_dict(), path + "/" + self.policy_name + "/" + "policy")
        torch.save(self.upd_queue.state_dict(), path + "/" + self.policy_name + "/" + "queue")

        return self.upd_policy, self.upd_queue

    def training_per_buff(self):
        i = 0
        queue_loss = None
        policy_loss = None
        self.base_queue.load_state_dict(self.upd_queue.state_dict())
        self.base_queue.eval()

        self.m_i = 10
        while i < self.m_i:
            # print(i)
            n_p_s, n_a, n_s, n_r, n_d, sk_idx = next(iter(self.dataloader))

            t_p_s = torch.tensor(n_p_s, dtype=torch.float32).to(self.device)
            t_a = torch.tensor(n_a, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)

            # policy_loss = torch.mean(torch.log(t_p_weight) - t_p_qvalue)
            # we already sampled according to policy
            i = 0
            policy_loss = torch.tensor(0).to(self.device).type(torch.float32)
            while i < 10:
                mean, cov, action = self.NAF_policy.prob(t_p_s)
                with torch.no_grad():
                    _action = action.cpu().numpy()
                    _action = torch.from_numpy(_action).to(self.device)
                    sa_pair = torch.cat((t_p_s, _action), -1).type(torch.float32)

                    target = self.base_queue(sa_pair)
                diff = (action - mean).unsqueeze(-1)
                prob = (-1/2)*torch.square(torch.transpose(diff, -1, -2)@torch.linalg.inv(cov)@diff)
                policy_loss += torch.sum(prob - target)
                i = i + 1

            t_trace = torch.tensor(n_d, dtype=torch.float32).to(self.device).unsqueeze(-1)

            t_a = torch.from_numpy(n_a).to(self.device)
            sa_pair = torch.cat((t_p_s, t_a), -1).type(torch.float32)
            t_p_qvalue = self.upd_queue(sa_pair)
            with torch.no_grad():
                n_a_expect = self.policy.select_action(n_s, sk_idx, per_one=0)
                t_a_expect = torch.tensor(n_a_expect, dtype=torch.float32).to(self.device)
                t_s = torch.tensor(n_s, dtype=torch.float32).to(self.device)
                new_sa_pair = torch.cat((t_s, t_a_expect), -1)
                with torch.no_grad():
                    t_qvalue = self.base_queue(new_sa_pair)
                    t_qvalue = t_qvalue*(GAMMA**t_trace) + t_r.unsqueeze(-1)

            queue_loss = self.criterion(t_p_qvalue, t_qvalue)

            self.optimizer_p.zero_grad()
            policy_loss.backward(retain_graph=True)
            for param in self.upd_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_p.step()

            self.optimizer_q.zero_grad()
            queue_loss.backward()
            for param in self.upd_queue.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            i = i + 1
        print("loss1 = ", policy_loss.squeeze())
        print("loss2 = ", queue_loss.squeeze())

        return torch.stack((policy_loss.squeeze(), queue_loss.squeeze()))
