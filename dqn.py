"""
Created on 2019-02-10 9:16 PM

@author: jack.lingheng.meng
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym

import argparse
import random
import pprint as pp

class qnetwork:
    def __init__(self, input_shape=[None, 4], action_size=2, scope=None):
        with tf.variable_scope(scope):
            self.scope = scope
            self.input_shape = input_shape
            self.action_size = action_size

            # Placeholders
            self.states = tf.placeholder(shape=input_shape, dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)

            powerful_qnetwork = True
            if powerful_qnetwork == True:
                # Fully connected layers
                fc1 = slim.fully_connected(self.states, 400, activation_fn=tf.nn.relu)
                fc2 = slim.fully_connected(fc1, 400, activation_fn=tf.nn.relu)
            else:
                # Fully connected layers
                fc1 = slim.fully_connected(self.states, 256, activation_fn=tf.nn.relu)
                fc2 = slim.fully_connected(fc1, 256, activation_fn=tf.nn.relu)

            self.q = slim.fully_connected(fc2, action_size, activation_fn=None)

            # Loss function
            self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
            self.responsible_output = tf.reduce_sum(tf.multiply(self.q, self.actions_onehot), axis=1, keep_dims=False)
            self.loss = tf.reduce_mean(tf.square(self.responsible_output - self.target_q))

            # Optimizer
            self.update_model = tf.train.AdamOptimizer().minimize(self.loss)

    def act(self, sess, state):
        q = sess.run(self.q, feed_dict={self.states: state[np.newaxis, ...]})
        return np.argmax(q)

    def train(self, sess, batch, discount_factor, tnet):
        assert len(batch) > 0
        states = np.vstack(batch[:, 0])
        actions = np.array(batch[:, 1])
        rewards = batch[:, 2]
        next_states = np.vstack(batch[:, 3])
        dones = batch[:, 4]

        next_q = sess.run(tnet.q, feed_dict={tnet.states: next_states})
        next_q = rewards + (1. - dones.astype(np.float32)) * discount_factor * np.amax(next_q, axis=1, keepdims=False)

        sess.run(self.update_model, feed_dict={self.states: states, self.actions: actions, self.target_q: next_q})
        return next_q

class Memory:
    def __init__(self, size):
       self.max_size = size
       self.mem = []

    def add(self, element):
        self.mem.append(element)

        if len(self.mem) > self.max_size:
            self.mem.pop(0)

    def sample(self, size):
        size = min(size, len(self.mem))
        return random.sample(self.mem, size)

def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def main(args):

    env = gym.make(args.env)
    args.action_size = env.action_space.n
    args.input_shape = [None] + list(env.observation_space.shape)

    # Epsilon parameter
    epsilon = args.epsilon_max

    # Replay memory
    memory = Memory(args.replay_mem_size)

    # Time step
    time_step = 0.

    # Initialize the agent
    qnet = qnetwork(input_shape=args.input_shape, action_size=args.action_size, scope='qnet')
    tnet = qnetwork(input_shape=args.input_shape, action_size=args.action_size, scope='tnet')
    update_ops = update_target_graph('qnet', 'tnet')

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

        for epoch in range(args.max_episodes):
            ep_reward = 0
            ep_ave_max_q = 0
            ep_steps = 0
            state = env.reset()
            while True:
                if args.render_env:
                    env.render()

                if np.random.rand() < epsilon:
                    action = np.random.randint(args.action_size)
                else:
                    action = qnet.act(sess, state)

                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                ep_steps += 1

                # Add to memory
                memory.add([state, action, reward, next_state, done])

                # Reduce epsilon
                time_step += 1.
                epsilon = args.epsilon_min + (args.epsilon_max - args.epsilon_min) * np.exp(-args.epsilon_decay * time_step)

                # Training step
                batch = np.array(memory.sample(args.batch_size))
                predicted_q_value = qnet.train(sess, batch, args.discount_factor, tnet)
                ep_ave_max_q += np.amax(predicted_q_value)
                # s <- s'
                state = np.copy(next_state)

                # Update target network
                if int(time_step) % args.target_update_freq == 0:
                    sess.run(update_ops)

                ep_steps += 1

                # if terminal or reach maximum length
                if done:
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / float((ep_steps + 1))
                    })

                    writer.add_summary(summary_str, epoch)
                    writer.flush()

                    print('| Episode: {0} | Steps: {1} | Reward: {2:.4f} | Qmax: {3:.4f}'.format(epoch,
                                                                                                 (ep_steps + 1),
                                                                                                 ep_reward,
                                                                                                 (ep_ave_max_q / float( ep_steps + 1))))
                    break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v0')
    parser.add_argument("--render-env", action="store_true")
    parser.add_argument("--action-size", type=int, default=2)
    parser.add_argument("--input-shape", type=list, default=[None, 4])
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument("--epsilon-max", type=float, default=1.)
    parser.add_argument("--epsilon-min", type=float, default=.01)
    parser.add_argument("--epsilon-decay", type=float, default=.001)
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default='./results/dqn/CartPole-v0/run1')

    parser.add_argument("--discount-factor", type=float, default=.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument('--max-episodes', type=int, default=50000, help='max num of episodes to do while training')
    parser.add_argument("--max-episode-len", type=int, default=1000, help='max length of 1 episode')

    parser.add_argument("--replay-mem-size", type=int, default=1000000)
    args = parser.parse_args()
    pp.pprint(args)

    main(args)

    # python dqn.py --env CartPole-v1 --summary-dir ./results/dqn/CartPole-v1/run1 --render-env True
    # python dqn.py --env Acrobot-v1 --summary-dir ./results/dqn/Acrobot-v1/run1 --render-env True