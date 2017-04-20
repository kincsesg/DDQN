import tensorflow as tf


class DQN:
    def __init__(self, params, name, TB_logpath):
        self.network_type = 'nature'
        self.params = params
        self.network_name = name
        self.x = tf.placeholder('float32', [None, 84, 84, 4], name=self.network_name + '_x')  # input
        self.q_t = tf.placeholder('float32', [None], name=self.network_name + '_q_t')  # input
        self.actions = tf.placeholder("float32", [None, params['num_act']], name=self.network_name + '_actions')  # input
        self.rewards = tf.placeholder("float32", [None], name=self.network_name + '_rewards')  # input
        self.terminals = tf.placeholder("float32", [None], name=self.network_name + '_terminals')  # input

        # conv1
        layer_name = 'conv1'
        size = 8
        channels = 4
        filters = 32
        stride = 4
        self.w1 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='VALID', name=self.network_name + '_' + layer_name + '_convs')
        self.o1 = tf.nn.relu(tf.add(self.c1, self.b1), name=self.network_name + '_' + layer_name + '_activations')

        # conv2
        layer_name = 'conv2'
        size = 4
        channels = 32
        filters = 64
        stride = 2
        self.w2 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='VALID', name=self.network_name + '_' + layer_name + '_convs')
        self.o2 = tf.nn.relu(tf.add(self.c2, self.b2), name=self.network_name + '_' + layer_name + '_activations')

        # conv3
        layer_name = 'conv3'
        size = 3
        channels = 64
        filters = 64
        stride = 1
        self.w3 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c3 = tf.nn.conv2d(self.o2, self.w3, strides=[1, stride, stride, 1], padding='VALID', name=self.network_name + '_' + layer_name + '_convs')
        self.o3 = tf.nn.relu(tf.add(self.c3, self.b3), name=self.network_name + '_' + layer_name + '_activations')

        # flat
        o3_shape = self.o3.get_shape().as_list()

        # fc3

        layer_name = 'fc4'
        hiddens = 512
        dim = o3_shape[1] * o3_shape[2] * o3_shape[3]
        self.o3_flat = tf.reshape(self.o3, [-1, dim], name=self.network_name + '_' + layer_name + '_input_flat')
        self.w4 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases')
        self.ip4 = tf.add(tf.matmul(self.o3_flat, self.w4), self.b4, name=self.network_name + '_' + layer_name + '_ips')
        self.o4 = tf.nn.relu(self.ip4, name=self.network_name + '_' + layer_name + '_activations')

        # fc4
        layer_name = 'fc5'
        hiddens = params['num_act']
        dim = 512
        self.w5 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b5 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases')
        self.y = tf.add(tf.matmul(self.o4, self.w5), self.b5, name=self.network_name + '_' + layer_name + '_outputs')

        # Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0 - self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Qxa = tf.multiply(self.y, self.actions)
        self.Q_pred = tf.reduce_max(self.Qxa, reduction_indices=1)
        self.diff = tf.subtract(self.yj, self.Q_pred)

        if self.params['clip_delta'] > 0:
            self.quadratic_part = tf.minimum(tf.abs(self.diff), tf.constant(self.params['clip_delta']))
            self.linear_part = tf.subtract(tf.abs(self.diff), self.quadratic_part)
            self.diff_square = 0.5 * tf.pow(self.quadratic_part, 2) + self.params['clip_delta'] * self.linear_part
        else:
            self.diff_square = tf.multiply(tf.constant(0.5), tf.pow(self.diff, 2))

        if self.params['batch_accumulator'] == 'sum':
            self.cost = tf.reduce_sum(self.diff_square)  # output
        else:
            self.cost = tf.reduce_mean(self.diff_square)  # output

        self.global_step = tf.Variable(0, name='global_step', trainable=False)  # output
        self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'], self.params['rms_decay'], 0.95, self.params['rms_eps']).minimize(self.cost, global_step=self.global_step)  # output

        # For tensorboard
        self.intra_ep_tr_rew = tf.placeholder(tf.float32, shape=(), name=self.network_name + '_tr_rew')  # input
        self.intra_ep_tr_q = tf.placeholder(tf.float32, shape=(), name=self.network_name + '_tr_q')  # input
        self.intra_ep_tr_cost = tf.placeholder(tf.float32, shape=(), name=self.network_name + '_tr_cost')  # input
        self.intra_ep_ev_rew = tf.placeholder(tf.float32, shape=(), name=self.network_name + '_ev_rew')  # input
        self.intra_ep_ev_q = tf.placeholder(tf.float32, shape=(), name=self.network_name + '_ev_q')  # input

        self.sum_tr_rew = tf.summary.scalar('Train_reward', self.intra_ep_tr_rew)
        self.sum_tr_q = tf.summary.scalar('Train_Q', self.intra_ep_tr_q)
        self.sum_tr_cost = tf.summary.scalar('Train_cost', self.intra_ep_tr_cost)
        self.sum_ev_rew = tf.summary.scalar('Eval_reward', self.intra_ep_ev_rew)
        self.sum_ev_q = tf.summary.scalar('Eval_Q', self.intra_ep_ev_q)

        self.merged_train = tf.summary.merge([self.sum_tr_rew,self.sum_tr_q,self.sum_tr_cost])
        self.merged_eval = tf.summary.merge([self.sum_ev_rew,self.sum_ev_q])

        self.TB_writer = tf.summary.FileWriter(TB_logpath, graph=tf.get_default_graph())