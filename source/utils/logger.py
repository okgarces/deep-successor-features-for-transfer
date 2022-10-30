import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time

import numpy as np

class Logger(object):
    def __init__(self):
        # Change logdir todo Data
        self.writer = SummaryWriter(f'data/dynamics_sfdqn_run_%s' % time.strftime('%d_%m_%Y_%H_%M_%S'))

    def log_scalar(self, category, value, epoch_iteration = None):
        self.writer.add_scalar(category, value, epoch_iteration)

    def log_scalars(self, category, value, global_step = None):
        self.writer.add_scalars(category, value, global_step)

    def log_histogram(self, category, values):
        self.writer.add_histogram(category, values)
    
    ###### Per task
    def log_progress(self, progress):
        task_id = progress.get('task') + 1
        self.log_scalar(f'Rewards/Episode/Task_{task_id}', progress.get('ep_reward'), progress.get('episodes'))
        self.log_scalar('GPI_%/Task', progress.get('GPI%'), task_id)
        self.log_scalar(f'Rewards/Step/Task_{task_id}', progress.get('reward'), progress.get('steps'))
        self.log_scalar(f'W_Error/Step/Task_{task_id}', progress.get('w_err'), progress.get('steps'))

        # History
        history = progress.get('reward_hist')
        history_accum = progress.get('cum_reward_hist')
        if len(history) > 0 and len(history_accum) > 0:
            self.log_histogram('History/Average_Reward', np.array(history))
            self.log_histogram('History/Accum_Reward', np.array(history_accum))

    def log_tasks_performance(self, performances):
        for task, performance in enumerate(performances):
            self.log_scalar('Overall_Performance/Task', performance, task + 1)

    ###### Per training steps
    def log_average_reward(self, progress, training_steps):
        self.log_scalar(f'Average_Reward/timesteps', progress, training_steps)

    def log_accumulative_reward(self, progress, training_steps):
        self.log_scalar(f'Accumulative_Reward/timesteps', progress, training_steps)

    def finalize(self):
        self.writer.flush()
        self.writer.close()

logger = Logger()
