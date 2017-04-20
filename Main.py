import sys
sys.path.append('/usr/local/lib/python3.5/site-packages')
from History import *
from DQN import *
import gym as gym
import tensorflow as tf
import numpy as np
import time
import cv2
import gc  # garbage colloector
import os
import shutil

gc.enable()

params = {
    'session_name' : '1', # name of the current execution
    'network_type': 'DDQN', # learning method/network type
    'ckpt_file': None,  # checkpoint file for save-restore functionality
    'save_dir': '/home/tr3zsor/Work/Dipterv2/repos/DDQN/ckpt', # checkpoint directory
    'save_fname':'model.ckpt', # checkpoint filename
    'TB_logpath': '/home/tr3zsor/Work/Dipterv2/DDQN/repos/TB_log', # path for TensorBoaard log file
    'environment': 'Breakout-v3',  # used game/environment
    'steps_per_epoch': 100000,  # steps during an epoch
    'num_epochs': 150,  # number of epochs
    'eval_freq': 20000,  # the frequency of the evaluation in steps
    'steps_per_eval': 10000,  # steps per evaluation cycles
    'intra_epoch_log_rate': 10,  # logging frequency for stepwise statistics in steps
    'TB_count_train': 0,  # summary index for Tensorboard
    'TB_count_eval': 0,  # summary index for Tensorboard
    'copy_freq': 10000,  # frequency of network copying in steps (tragetnet <- qnet), if 0, only qnet is used
    'db_size': 1000000,  # number of observation stored in the replay memory
    'batch': 32,  # size of a batxh during network training
    'num_act': 0,  # the number of possible steps in a game
    'net_train': 10,  # network retrain frequency in steps
    'eps': 1.0,  # the initial (and actual) epsilon value for epsilon greedy strategy
    'eps_min': 0.1,  # the value of epsilon cannot decreease below this value during the training
    'eps_eval': 0.05,  # the epsilon value used for evaluation
    'discount': 0.99,  # discount factor
    'lr': 0.00025,  # learning rate
    'rms_decay': 0.99,  # decay parameter for rmsprop algorithm
    'rms_eps': 1e-6,  # epsilon parameter for rmsprop algorithm
    'train_start': 10000,  # the number of steps at the beginnig of the training while the agent does nothing, but collects observations through random wandering
    'img_scale': 255.0,  # the scaling (0..1) variable for grayscale pictures
    'clip_delta': 0,
    'gpu_fraction': 0.85,  # gpu memory fraction used by tensorflow
    'batch_accumulator': 'mean',  # the method for accumulation the loss for batches during the training of the network
}

class double_DQN:
    def __init__(self, params):
        print('Initializing Module...')
        self.params = params

        self.gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction = self.params['gpu_fraction']))

        self.sess = tf.Session(config=self.gpu_config)
        self.DB = History(self.params)
        self.engine = gym.make(self.params['environment'])
        self.params['num_act'] = self.engine.action_space.n

        # region Checking for checkpoint
        if os.path.exists(self.params['save_dir']):
            self.params['ckpt_file'] = self.params['save_dir'] + '/' + self.params['save_fname']
        # endregion

        self.training = True
        self.hist_collecting = True
        self.net_retraining = False
        self.loaded = True
        self.build_net()

        # region Initializing log variables
        # by epoch statistics
        if self.params['ckpt_file'] is None:
            self.by_ep_maxrew_tr = 0
            self.by_ep_sumrew_tr = 0
            self.by_ep_maxQ_tr = 0
            self.by_ep_sumQ_tr = 0
            self.by_ep_maxcost_tr = 0
            self.by_ep_sumcost_tr = 0
            self.by_ep_maxrew_ev = 0
            self.by_ep_sumrew_ev = 0
            self.by_ep_maxQ_ev = 0
            self.by_ep_sumQ_ev = 0

        # final statistics
        self.total_maxrew_tr = 0
        self.total_sumrew_tr = 0
        self.total_maxQ_tr = 0
        self.total_sumQ_tr = 0
        self.total_maxcost_tr = 0
        self.total_sumcost_tr = 0
        self.total_maxrew_ev = 0
        self.total_sumrew_ev = 0
        self.total_maxQ_ev = 0
        self.total_sumQ_ev = 0

        self.reset_statistics('all')
        self.train_cnt = self.sess.run(self.qnet.global_step)
        # endregion

        # region Initializing the inner variables of the algorithm
        self.state_proc = np.zeros((84, 84, 4))
        self.action = -1
        self.terminal = False
        self.reward = 0
        self.state = self.engine.reset()
        self.engine.render()
        self.state_resized = cv2.resize(self.state, (84, 110))
        self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
        self.state_gray_old = None
        self.state_proc[:, :, 3] = self.state_gray[26:110, :] / self.params['img_scale']

        self.cost = 0.0

        if self.params['ckpt_file'] is None:
            self.step = 0
            self.steps_train = 0
        self.steps_eval = 0
        # endregion

    def build_net(self):
        print('Building QNet and targetnet...')
        self.qnet = DQN(self.params, 'qnet', self.params['TB_logpath'])
        self.targetnet = DQN(self.params, 'targetnet', self.params['TB_logpath'])
        self.sess.run(tf.global_variables_initializer())
        saver_dict = {'qw1': self.qnet.w1, 'qb1': self.qnet.b1,
                      'qw2': self.qnet.w2, 'qb2': self.qnet.b2,
                      'qw3': self.qnet.w3, 'qb3': self.qnet.b3,
                      'qw4': self.qnet.w4, 'qb4': self.qnet.b4,
                      'qw5': self.qnet.w5, 'qb5': self.qnet.b5,
                      'tw1': self.targetnet.w1, 'tb1': self.targetnet.b1,
                      'tw2': self.targetnet.w2, 'tb2': self.targetnet.b2,
                      'tw3': self.targetnet.w3, 'tb3': self.targetnet.b3,
                      'tw4': self.targetnet.w4, 'tb4': self.targetnet.b4,
                      'tw5': self.targetnet.w5, 'tb5': self.targetnet.b5,
                      'step': self.qnet.global_step}
        self.saver = tf.train.Saver(saver_dict)
        self.cp_ops = [
            self.targetnet.w1.assign(self.qnet.w1), self.targetnet.b1.assign(self.qnet.b1),
            self.targetnet.w2.assign(self.qnet.w2), self.targetnet.b2.assign(self.qnet.b2),
            self.targetnet.w3.assign(self.qnet.w3), self.targetnet.b3.assign(self.qnet.b3),
            self.targetnet.w4.assign(self.qnet.w4), self.targetnet.b4.assign(self.qnet.b4),
            self.targetnet.w5.assign(self.qnet.w5), self.targetnet.b5.assign(self.qnet.b5)]
        self.sess.run(self.cp_ops)

        if self.params['ckpt_file'] is not None:
            print('\x1b[1;30;41m RUN LOAD \x1b[0m')
            self.load()

        print('Networks had been built!')
        sys.stdout.flush()

    def start(self):

        # region Creating tables for logging
        # intra epoch tables
        if self.params['ckpt_file'] is not None:
            try:
                self.log_intra_epoch_train = open('intra_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'a')
            except:
                self.log_intra_epoch_train = open('intra_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_intra_epoch_train.write('epoch,step,reward,Q,time\n')
            try:
                self.log_intra_epoch_eval = open('intra_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'a')
            except:
                self.log_intra_epoch_eval = open('intra_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_intra_epoch_eval.write('epoch,eval_step,reward,Q,time\n')
        else:
            self.log_intra_epoch_train = open('intra_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_intra_epoch_train.write('epoch,step,reward,Q,time\n')

            self.log_intra_epoch_eval = open('intra_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_intra_epoch_eval.write('epoch,eval_step,reward,Q,time\n')

        # by epoch tables
        if self.params['ckpt_file'] is not None:
            try:
                self.log_by_epoch_train = open('by_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'a')
            except:
                self.log_by_epoch_train = open('by_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_by_epoch_train.write('epoch,step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,maxcost,sumcost,avgcost,time\n')
            try:
                self.log_by_epoch_eval = open('by_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'a')
            except:
                self.log_by_epoch_eval = open('by_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_by_epoch_eval.write('epoch,step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,time\n')
        else:
            self.log_by_epoch_train = open('by_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_by_epoch_train.write('epoch,step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,maxcost,sumcost,avgcost,time\n')

            self.log_by_epoch_eval = open('by_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_by_epoch_eval.write('epoch,step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,time\n')

        # final tables
        if self.params['ckpt_file'] is not None:
            try:
                self.log_total_train = open('total_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'a')
            except:
                self.log_total_train = open('total_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_total_train.write('epochs,steps_per_epoch,hist_size,tr_step,eval_step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,maxcost,sumcost,avgcost,time\n')
            try:
                self.log_total_eval = open('total_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'e')
            except:
                self.log_total_eval = open('total_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_total_eval.write('epochs,steps_per_epoch,hist_size,tr_step,eval_step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,time\n')
        else:
            self.log_total_train = open('total_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_total_train.write('epochs,steps_per_epoch,hist_size,tr_step,eval_step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,maxcost,sumcost,avgcost,time\n')

            self.log_total_eval = open('total_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_total_eval.write('epochs,steps_per_epoch,hist_size,tr_step,eval_step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,time\n')
        # endregion

        self.start_time = time.time()
        print(self.params)
        print('Start training!')
        print('Collecting replay memory for ' + str(self.params['train_start']) + ' steps')

        # the algorithm itself is in this cycle
        while self.step <= self.params['steps_per_epoch'] * self.params['num_epochs']:
            
            # region Recalculating cycle_state variables, and reset step vars
            if self.DB.get_size() == self.params['train_start']:
                self.hist_collecting = False

            if self.training and self.loaded and not self.hist_collecting and self.steps_train > 0 and self.steps_train % self.params['net_train'] == 0:
                self.net_retraining = True
            else:
                self.net_retraining = False

            if self.training and self.loaded and not self.hist_collecting and self.steps_train > 0 and self.steps_train % self.params['eval_freq'] == 0:
                self.training = False
                self.steps_eval = 0

            if not self.training and not self.hist_collecting and self.steps_eval > 0 and self.steps_eval % self.params['steps_per_eval'] == 0:
                self.training = True
                self.save()

            if self.training and self.steps_train > 0 and self.steps_train % self.params['steps_per_epoch'] == 0:
                self.write_log_train('epoch_end')
                self.write_log_eval('epoch_end')
                self.reset_statistics('epoch_end')
                self.steps_train = 0

            if self.training and self.step > 0 and self.step % (self.params['steps_per_epoch'] * self.params['num_epochs']) == 0:
                self.write_log_train('final')
                self.write_log_eval('final')
                break

            if not self.loaded and not self.hist_collecting:
                self.loaded = True
            #endregion

            # region Increasing step variables
            if self.training:
                if not self.hist_collecting:
                    self.step += 1
                    self.steps_train += 1
            else:
                if not self.net_retraining:
                    self.steps_eval += 1
            # endregion

            # region Printing actual state
            if self.hist_collecting:
                print('History size: ' + str(self.DB.get_size()))
            if self.training and not self.hist_collecting:
                print('Epoch: ' + str(self.step // self.params['steps_per_epoch'] + 1) + ' Training step: ' + str(self.steps_train))
            if not self.training and not self.hist_collecting:
                print('Epoch: ' + str(self.step // self.params['steps_per_epoch'] + 1) + ' Eval. step: ' + str(self.steps_eval))
            sys.stdout.flush()
            # endregion


            if self.training or self.net_retraining:

                # region Stores the current [state,reward,action,terminal] tuple to the history
                if self.state_gray_old is not None and self.training:
                    self.DB.insert(self.state_gray_old[26:110, :], self.reward_scaled, self.action_idx, self.terminal)
                # endregion

                if self.hist_collecting:

                    # region Set epsilon for history collection
                    self.params['eps'] = 1.0
                    # endregion

                else:

                    # region Copy network at every <copy_freq> steps
                    if self.params['copy_freq'] > 0 and self.step % self.params['copy_freq'] == 0:
                        print('&&& Copying Qnet to targetnet\n')
                        self.sess.run(self.cp_ops)
                    # endregion

                    if self.net_retraining:

                        # region Retrain network
                        bat_s, bat_a, bat_t, bat_n, bat_r = self.DB.get_batches()
                        bat_a = self.get_onehot(bat_a)

                        if self.params['copy_freq'] > 0: # get predictions from the network for the input batches
                            feed_dict = {self.qnet.x: bat_n}
                            q_tmp = self.sess.run(self.qnet.y, feed_dict=feed_dict)
                            a_nextbest = np.argmax(q_tmp, axis=1)  # best action in the next state given by the q-network

                            feed_dict = {self.targetnet.x: bat_n}
                            q_t = self.sess.run(self.targetnet.y, feed_dict=feed_dict)
                        else:
                            feed_dict = {self.qnet.x: bat_n}
                            q_tmp = self.sess.run(self.qnet.y, feed_dict=feed_dict)
                            a_nextbest = np.argmax(q_tmp, axis=1)  # best action in the next state given by the q-network

                            feed_dict = {self.qnet.x: bat_n}
                            q_t = self.sess.run(self.qnet.y, feed_dict=feed_dict)

                        q_t = q_t[range(len(q_t)),a_nextbest]  # value of the action given by the q-network calculated by the target network

                        feed_dict = {self.qnet.x: bat_s, self.qnet.q_t: q_t, self.qnet.actions: bat_a, self.qnet.terminals: bat_t, self.qnet.rewards: bat_r}
                        _, self.train_cnt, self.cost = self.sess.run([self.qnet.rmsprop, self.qnet.global_step, self.qnet.cost], feed_dict=feed_dict)  # retraining

                        self.by_ep_sumcost_tr += np.sqrt(self.cost)
                        self.total_sumcost_tr += np.sqrt(self.cost)

                        if self.by_ep_maxcost_tr < np.sqrt(self.cost):
                            self.by_ep_maxcost_tr = np.sqrt(self.cost)

                        if self.total_maxcost_tr < np.sqrt(self.cost):
                            self.total_maxcost_tr = np.sqrt(self.cost)
                        print('Network retrained!')
                        # endregion

                    # region Decrease epsilon for training
                        self.params['eps'] = max(self.params['eps_min'], 1.0 - (1.0 - self.params['eps_min']) * (float(self.step) + float(self.params['steps_per_epoch'])) / (float(self.params['steps_per_epoch']) * float(self.params['num_epochs'])))
                    # endregion

                    # region Log training
                    self.write_log_train('in_epoch')
                    # endregion

            else:

                # region Set epsilon for eval
                self.params['eps'] = self.params['eps_eval']
                # endregion

                # region Log evaluation
                self.write_log_eval('in_epoch')
                # endregion

            # region Reset game if a terminal state reached
            if self.terminal:
                self.reset_game()
            # endregion

            # region Chose action
            self.action_idx, self.action, self.maxQ = self.select_action(self.state_proc)

            if not self.hist_collecting:
                if self.by_ep_maxQ_tr < self.maxQ:
                    self.by_ep_maxQ_tr = self.maxQ
                if self.total_maxQ_tr < self.maxQ:
                    self.total_maxQ_tr = self.maxQ

                if self.by_ep_maxQ_ev < self.maxQ:
                    self.by_ep_maxQ_ev = self.maxQ
                if self.total_maxQ_ev < self.maxQ:
                    self.total_maxQ_ev = self.maxQ

                self.by_ep_sumQ_tr += self.maxQ
                self.total_sumQ_tr += self.maxQ

                self.by_ep_sumQ_ev += self.maxQ
                self.total_sumQ_ev += self.maxQ
            # endregion

            # region Execute action, observe environment and scale reward
            self.state, self.reward, t, _ = self.engine.step(self.action)
            extr_rew = self.extract_reward(self.state)
            self.engine.render()
            self.terminal = int(t)
            #self.reward_scaled = self.reward // max(1, abs(self.reward))
            self.reward_scaled = self.reward * extr_rew
            if abs(self.reward) > 0.0:
                print('\x1b[1;30;41m' + str(self.reward) + '\x1b[0m')
                print('\x1b[1;30;41m' + str(self.reward_scaled) + '\x1b[0m')

            if not self.hist_collecting:
                if self.by_ep_maxrew_tr < self.reward_scaled:
                    self.by_ep_maxrew_tr = self.reward_scaled
                if self.total_maxrew_tr < self.reward_scaled:
                    self.total_maxrew_tr = self.reward_scaled

                if self.by_ep_maxrew_ev < self.reward_scaled:
                    self.by_ep_maxrew_ev = self.reward_scaled
                if self.total_maxrew_ev < self.reward_scaled:
                    self.total_maxrew_ev = self.reward_scaled

                self.by_ep_sumrew_tr += self.reward_scaled
                self.total_sumrew_tr += self.reward_scaled

                self.by_ep_sumrew_ev += self.reward_scaled
                self.total_sumrew_ev += self.reward_scaled
            # endregion

            # region s <- s'
            self.state_gray_old = np.copy(self.state_gray)
            self.state_proc[:, :, 0:3] = self.state_proc[:, :, 1:4]
            self.state_resized = cv2.resize(self.state, (84, 110))
            self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
            self.state_proc[:, :, 3] = self.state_gray[26:110, :] / self.params['img_scale']
            # endregion

            # TODO : add video recording

    def reset_game(self):
        self.state_proc = np.zeros((84, 84, 4))
        self.action = -1
        self.terminal = False
        self.reward = 0
        self.state = self.engine.reset()
        self.engine.render()
        self.state_resized = cv2.resize(self.state, (84, 110))
        self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
        self.state_gray_old = None
        self.state_proc[:, :, 3] = self.state_gray[26:110, :] / self.params['img_scale']

    def reset_statistics(self, state):
        if state == 'epoch_end':
            self.by_ep_maxrew_tr = 0
            self.by_ep_sumrew_tr = 0
            self.by_ep_maxQ_tr = 0
            self.by_ep_sumQ_tr = 0
            self.by_ep_maxcost_tr = 0
            self.by_ep_sumcost_tr = 0
            self.by_ep_maxrew_ev = 0
            self.by_ep_sumrew_ev = 0
            self.by_ep_maxQ_ev = 0
            self.by_ep_sumQ_ev = 0

    def select_action(self, st):
        if np.random.rand() > self.params['eps']:
            # greedy with random tie-breaking
            Q_pred = self.sess.run(self.qnet.y, feed_dict={self.qnet.x: np.reshape(st, (1, 84, 84, 4))})[0]
            a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
            if len(a_winner) > 1:
                act_idx = a_winner[np.random.randint(0, len(a_winner))][0]
                return act_idx, act_idx, np.amax(Q_pred)
            else:
                act_idx = a_winner[0][0]
                return act_idx, act_idx, np.amax(Q_pred)
        else:
            # random
            act_idx = np.random.randint(0, self.engine.action_space.n)
            Q_pred = self.sess.run(self.qnet.y, feed_dict={self.qnet.x: np.reshape(st, (1, 84, 84, 4))})[0]
            return act_idx, act_idx, Q_pred[act_idx]

    def get_onehot(self, actions):
        actions_onehot = np.zeros((self.params['batch'], self.params['num_act']))

        for i in range(self.params['batch']):
            actions_onehot[i, int(actions[i])] = 1

        return actions_onehot

    def extract_reward(self, state):
        if(self.params['environment'] == 'Breakout-v3'):
            state_resized = cv2.resize(state, (84, 110))
            state_gray = cv2.cvtColor(state_resized, cv2.COLOR_BGR2GRAY)
            extract_source =state_gray[30:49,4:80]
            return 1444 - np.count_nonzero(extract_source)
        else:
            return 1

    def save(self):
        directory = self.params['save_dir']
        filename = self.params['save_fname']
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('Saving checkpoint : ' + directory + '/' + filename)
        self.saver.save(self.sess, directory + '/' + filename)
        sys.stdout.write('$$$ Model saved : %s\n\n' % str(directory + '/' + filename))
        sys.stdout.flush()

        self.saved_stat = open(directory + '/model.csv', 'w')
        self.saved_stat.write('by_ep_maxrew_tr,by_ep_sumrew_tr,by_ep_maxQ_tr,by_ep_sumQ_tr,by_ep_maxcost_tr,by_ep_sumcost_tr,by_ep_maxrew_ev,'
                                         'by_ep_sumrew_ev,by_ep_maxQ_ev,by_ep_sumQ_ev\n')
        self.saved_stat.write(str(self.by_ep_maxrew_tr) + ',' + str(self.by_ep_sumrew_tr) + ',' + str(self.by_ep_maxQ_tr) + ',' + str(self.by_ep_sumQ_tr) + ',' +
                             str(self.by_ep_maxcost_tr) + ',' + str(self.by_ep_sumcost_tr) + ',' + str(self.by_ep_maxrew_ev) + ',' + str(self.by_ep_sumrew_ev) + ',' +
                             str(self.by_ep_maxQ_ev) + ',' + str(self.by_ep_sumQ_ev) + '\n')
        self.saved_stat.close()

    def load(self):
        print('Loading checkpoint : ' + self.params['ckpt_file'])
        self.saver.restore(self.sess, self.params['ckpt_file'])
        temp_train_cnt = self.sess.run(self.qnet.global_step)
        self.step = temp_train_cnt * self.params['net_train']
        self.steps_train = self.step % self.params['steps_per_epoch']
        self.steps_eval = self.params['steps_per_eval']
        self.loaded = False

        self.saved_stat = open(str.split(self.params['ckpt_file'],'.')[0] + '.csv', 'r')
        self.saved_stat.readline()
        stats = self.saved_stat.readline()
        splitted_list = stats.split(',')
        self.by_ep_maxrew_tr = float(splitted_list[0])
        self.by_ep_sumrew_tr = float(splitted_list[1])
        self.by_ep_maxQ_tr = float(splitted_list[2])
        self.by_ep_sumQ_tr = float(splitted_list[3])
        self.by_ep_maxcost_tr = float(splitted_list[4])
        self.by_ep_sumcost_tr = float(splitted_list[5])
        self.by_ep_maxrew_ev = float(splitted_list[6])
        self.by_ep_sumrew_ev = float(splitted_list[7])
        self.by_ep_maxQ_ev = float(splitted_list[8])
        self.by_ep_sumQ_ev = float(splitted_list[9])

    def write_log_train(self, cycle_state):
        epoch = self.step // self.params['steps_per_epoch'] + 1
        if cycle_state == 'in_epoch':
            if self.steps_train % self.params['intra_epoch_log_rate'] == 0:
                self.log_intra_epoch_train.write(str(epoch) + ',' + str(self.steps_train) + ',' + str(self.reward_scaled) + ',' +
                                                 str(self.maxQ) + ',' + str(time.time()) + '\n')
                self.log_intra_epoch_train.flush()

                # for Tensorboard
                feed_dict = {self.qnet.intra_ep_tr_rew: self.reward_scaled, self.qnet.intra_ep_tr_q: self.maxQ, self.qnet.intra_ep_tr_cost: self.cost}

                summary = self.sess.run(self.qnet.merged_train, feed_dict=feed_dict)
                self.qnet.TB_writer.add_summary(summary, self.params['TB_count_train'])
                self.params['TB_count_train'] += 1

            sys.stdout.write('Epoch : %d , Step : %d , Reward : %f, Q : %.3f , Time : %.1f\n' % (epoch,self.step,self.reward_scaled,self.maxQ,time.time()))
            sys.stdout.flush()

        if cycle_state == 'epoch_end':
            self.log_by_epoch_train.write(str(epoch) + ',' + str(self.step) + ',' +
                                             str(self.by_ep_maxrew_tr) + ',' + str(self.by_ep_sumrew_tr) + ',' + str(self.by_ep_sumrew_tr / self.params['steps_per_epoch']) + ',' +
                                             str(self.by_ep_maxQ_tr) + ',' + str(self.by_ep_sumQ_tr) + ',' + str(self.by_ep_sumQ_tr / self.params['steps_per_epoch']) + ',' +
                                             str(self.by_ep_maxcost_tr) + ',' + str(self.by_ep_sumcost_tr) + ',' + str(self.by_ep_sumcost_tr / self.params['steps_per_epoch']) + ',' +
                                             str(time.time()) + '\n')
            self.log_by_epoch_train.flush()

        if cycle_state == 'final':
            self.log_total_train.write(str(epoch) + ',' + str(self.params['steps_per_epoch']) + ',' + str(self.params['db_size']) + ',' +
                                       str(self.params['eval_freq']) + ',' + str(self.params['steps_per_eval']) + ',' +
                                       str(self.total_maxrew_tr) + ',' + str(self.total_sumrew_tr) + ',' + str(self.total_sumrew_tr / (self.params['steps_per_epoch'] * self.params['num_epochs'])) + ',' +
                                       str(self.total_maxQ_tr) + ',' + str(self.total_sumQ_tr) + ',' + str(self.total_sumQ_tr / (self.params['steps_per_epoch'] * self.params['num_epochs'])) + ',' +
                                       str(self.total_maxcost_tr) + ',' + str(self.total_sumcost_tr) + ',' + str(self.total_sumcost_tr / (self.params['steps_per_epoch'] * self.params['num_epochs'])) + ',' +
                                       str(time.time()) + '\n')
            self.log_total_train.flush()

    def write_log_eval(self, cycle_state):
        epoch = self.step // self.params['steps_per_epoch'] + 1
        if cycle_state == 'in_epoch':
            if self.steps_eval % self.params['intra_epoch_log_rate'] == 0:
                self.log_intra_epoch_eval.write(str(epoch) + ',' + str(self.steps_eval) + ',' + str(self.reward_scaled) + ',' +
                                                str(self.maxQ) + ',' + str(time.time()) + '\n')
                self.log_intra_epoch_eval.flush()

                # for Tensorboard
                feed_dict = {self.qnet.intra_ep_ev_rew: self.reward_scaled, self.qnet.intra_ep_ev_q: self.maxQ}

                summary = self.sess.run(self.qnet.merged_eval, feed_dict=feed_dict)
                self.qnet.TB_writer.add_summary(summary, self.params['TB_count_eval'])
                self.params['TB_count_eval'] += 1

            sys.stdout.write(
                'Epoch : %d , Eval.Step : %d , Reward : %f, Q : %.3f , Time : %.1f\n' % (epoch, self.steps_eval, self.reward_scaled, self.maxQ, time.time()))
            sys.stdout.flush()

        if cycle_state == 'epoch_end':
            self.log_by_epoch_eval.write(str(epoch) + ',' + str(self.step) + ',' +
                                            str(self.by_ep_maxrew_ev) + ',' + str(self.by_ep_sumrew_ev) + ',' + str(self.by_ep_sumrew_ev / self.params['steps_per_epoch']) + ',' +
                                            str(self.by_ep_maxQ_ev) + ',' + str(self.by_ep_sumQ_ev) + ',' + str(self.by_ep_sumQ_ev / self.params['steps_per_epoch']) + ',' +
                                            str(time.time()) + '\n')
            self.log_by_epoch_eval.flush()

        if cycle_state == 'final':
            self.log_total_eval.write(str(epoch) + ',' + str(self.params['steps_per_epoch']) + ',' + str(self.params['db_size']) + ',' +
                                       str(self.params['eval_freq']) + ',' + str(self.params['steps_per_eval']) + ',' +
                                       str(self.total_maxrew_ev) + ',' + str(self.total_sumrew_ev) + ',' + str(self.total_sumrew_ev / (self.params['steps_per_epoch'] * self.params['num_epochs'])) + ',' +
                                       str(self.total_maxQ_ev) + ',' + str(self.total_sumQ_ev) + ',' + str(self.total_sumQ_ev / (self.params['steps_per_epoch'] * self.params['num_epochs'])) + ',' +
                                       str(time.time()) + '\n')
            self.log_total_eval.flush()

# entry point
if __name__ == "__main__":
    ddqn = double_DQN(params)
    ddqn.start()