import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from abc import abstractmethod
import time

logger = None

class LoggerBase(object):

    def __init__(self, debug=True):
        self.debug = debug

    @abstractmethod
    def log_scalar(self, category, value, epoch_iteration = None):
        raise NotImplemented()

    @abstractmethod
    def log_scalars(self, category, value, global_step = None):
        raise NotImplemented()

    @abstractmethod
    def log_histogram(self, category, values):
        raise NotImplemented()

    ###### Per task
    def log_progress(self, progress):
        task_id = progress.get('task') + 1
        self.log_scalar(f'Rewards/Episode/Task_{task_id}', progress.get('ep_reward'), progress.get('episodes'))
        self.log_scalar('GPI_%/Task', progress.get('GPI%'), task_id)
        self.log_scalar(f'Rewards/Step/Task_{task_id}', progress.get('reward'), progress.get('steps'))
        self.log_scalar(f'W_Error/Step/Task_{task_id}', progress.get('w_err'), progress.get('steps'))

        # History
        #history = progress.get('reward_hist')
        #history_accum = progress.get('cum_reward_hist')
        #if len(history) > 0 and len(history_accum) > 0:
        #    self.log_histogram('History/Average_Reward', np.array(history))
        #    self.log_histogram('History/Accum_Reward', np.array(history_accum))

    def log_target_error_progress(self, progress):
        task_id = progress.get('task') + 1
        steps = progress.get('steps')
        self.log_scalar(f'Target_Tasks/W_Error/Ev_Steps/task_{task_id}', progress.get('w_error'), steps)
        self.log_scalar(f'Target_Tasks/Rewards/Ev_Steps/task_{task_id}', progress.get('reward'), steps)

        target_phi_loss = progress.get('phi_loss')
        target_psi_loss = progress.get('psi_loss')
        target_loss_coefficient = progress.get('target_loss_coefficient')

        if target_phi_loss is not None or target_psi_loss is not None or target_loss_coefficient is not None:
            self.log_scalar(f'Target_Tasks/Phi_Loss/Ev_Steps/task_{task_id}', target_phi_loss, steps)
            self.log_scalar(f'Target_Tasks/Psi_Loss/Ev_Steps/task_{task_id}', target_psi_loss, steps)
            self.log_scalar(f'Target_Tasks/Losses/Coefficients/Ev_Steps/task_{task_id}', target_loss_coefficient, steps)


    def log_tasks_performance(self, performances):
        for task, performance in enumerate(performances):
            self.log_scalar('Overall_Performance/Task', performance, task + 1)

    ###### Per training steps
    def log_average_reward(self, progress, training_steps):
        self.log_scalar(f'Average_Reward/timesteps', progress, training_steps)

    def log_accumulative_reward(self, progress, training_steps):
        self.log_scalar(f'Accumulative_Reward/timesteps', progress, training_steps)

    def log_phi_loss(self, progress, training_steps):
        self.log_scalar(f'Losses/Phi_Loss/timesteps', progress, training_steps)

    def log_psi_loss(self, progress, training_steps):
        self.log_scalar(f'Losses/Psi_Loss/timesteps', progress, training_steps)

    def log_total_loss(self, progress, training_steps):
        self.log_scalar(f'Losses/Total_Loss/timesteps', progress, training_steps)

    def log_loss_coefficient(self, progress, training_steps):
        if len(progress) > 1:
            self.log_scalar(f'Losses/Coefficients_L1/timesteps', progress[0], training_steps)
            self.log_scalar(f'Losses/Coefficients_L2/timesteps', progress[1], training_steps)
        else:
            self.log_scalar(f'Losses/Coefficients/timesteps', progress[0], training_steps)

    def log_losses(self, total_loss, psi_loss, phi_loss, loss_coefficient, training_steps):
        self.log_phi_loss(phi_loss, training_steps)
        self.log_psi_loss(psi_loss, training_steps)
        self.log_total_loss(total_loss, training_steps)
        self.log_loss_coefficient(loss_coefficient, training_steps)

    def log_omegas_learning_rate(self, learning_rate, task_id, total_steps):
        self.log_scalar(f'Target_Tasks/Omegas_Learning_Rate/Ev_Steps/task_{task_id + 1}', learning_rate, total_steps)

    def log_source_performance(self, task_id, reward, training_steps):
        self.log_scalar(f'Source_Tasks/Rewards/task_{task_id + 1}', reward, training_steps)

    def finalize(self):
        raise NotImplemented()


class Logger(LoggerBase):
    def __init__(self, debug=True):
        super().__init__(debug)
        # Change logdir todo Data
        self.writer = SummaryWriter(f'data/dynamics_sfdqn_run_%s' % time.strftime('%d_%m_%Y_%H_%M_%S'))

    def log_scalar(self, category, value, epoch_iteration = None):
        self.writer.add_scalar(category, value, epoch_iteration)

    def log_scalars(self, category, value, global_step = None):
        self.writer.add_scalars(category, value, global_step)

    def log_histogram(self, category, values):
        self.writer.add_histogram(category, values)

    def finalize(self):
        self.writer.flush()
        self.writer.close()

class MockLogger(LoggerBase):

    def log_scalar(self, category, value, epoch_iteration = None):
        print(f'{(category)} Value: {value} Epoch: {epoch_iteration}')

    def log_scalars(self, category, value, global_step = None):
        print(f'{(category)} Value: {value} Epoch: {global_step}')

    def log_histogram(self, category, values):
        print(f'{(category)} Values: {values}')

    def finalize(self):
        pass

############################33
from time import gmtime, strftime
import numpy as np
import pandas as pd
import json

class LoggerCSV:
    SOURCE_TASK = 'source'
    TARGET_TASK = 'target'

    HEADERS = ['task_id', 'reward', 'step', 'accum_loss', 'q_loss', 'psi_loss', 'phi_loss']

    def __init__(self, root_path, prefix=None):
        super().__init__()
        prefix = prefix+'_' if prefix is not None else ''
        self.source_tasks_file = f'{root_path}results/{prefix}source_performance_{strftime("%d_%b_%Y_%H_%M_%S", gmtime())}.csv'
        self.target_tasks_file = f'{root_path}results/{prefix}target_performance_{strftime("%d_%b_%Y_%H_%M_%S", gmtime())}.csv'
        self.log_task_file = f'{root_path}results/{prefix}log_performance_{strftime("%d_%b_%Y_%H_%M_%S", gmtime())}.csv'

    def log(self, log_dictionary):
        filename = self.log_task_file
        with open(filename, 'a') as f:
            f.write(json.dumps(str(log_dictionary)) + '\n')
            # np.savetxt(f, json.dumps(log_dictionary), delimiter=',', newline='\n')

    def log_agent_performance(self, task, reward, step, accum_loss, *args, **kwargs):
        values = np.array([task, reward, step, accum_loss, *args])
        type_task = kwargs.get('type_task', self.SOURCE_TASK)
        filename = self.source_tasks_file if type_task == self.SOURCE_TASK else self.target_tasks_file

        with open(filename, 'a') as f:
            np.savetxt(f, np.column_stack(values), delimiter=',', newline='\n')

    def load_text(self, type_task='source'):
        filename = self.source_tasks_file if type_task == self.SOURCE_TASK else self.target_tasks_file

        return pd.DataFrame(np.loadtxt(filename, delimiter=','))


def set_logger_level(use_logger=False, root_path='', prefix=None):
    global logger
    if logger is None:
        logger = LoggerCSV(root_path, prefix) if use_logger else MockLogger()
        return logger
    return logger

def get_logger_level():
    return logger