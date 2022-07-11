from control.base import BasePolicy
import torch
from my_model import Network, policy
from utils import dataset, dataloader, buffer
import trainer
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DQNPolicy(BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.updatedDQN = Network.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.baseDQN = Network.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.updatedDQN.load_state_dict(torch.load(self.PARAM_PATH_TEST))
        self.baseDQN.load_state_dict(self.updatedDQN.state_dict())
        self.baseDQN.eval()
        self.DQN_policy = policy.Policy(self.cont, self.baseDQN, self.env_n)
        self.buffer = buffer.Simulate(self.env, self.DQN_policy)
        self.data = dataset.SimData(capacity=self.ca)
        self.buffer.renewal_memory(self.ca, self.data)
        self.dataloader = dataloader.CustomDataLoader(self.data, batch_size=BATCH_SIZE)
        self.train = trainer.Train(self.env_n, self.dataloader, self.cont)
        self.optimizer = torch.optim.SGD(self.updatedDQN.parameters(), lr=self.lr)
        self.writer = SummaryWriter('RLresult/' + self.env_n + '/' + self.cont)

    def training(self):
        i = 0
        while i < self.t_i:
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data)
            loss = self.train.training(self.t_i, self.b_s, self.optimizer, self.updatedDQN, self.baseDQN)
            writer.add_scalar("loss", loss, i)
            torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH_TEST)
            self.baseDQN.load_state_dict(self.updatedDQN.state_dict())
            self.baseDQN.eval()

        self.env.close()
        writer.flush()
        writer.close()
