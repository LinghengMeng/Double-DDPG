""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import os
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer

import logging

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, continuous_act_space_flag, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.continuous_act_space_flag = continuous_act_space_flag
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)

        # If Actor acts on discrete action space, use Softmax
        if self.continuous_act_space_flag is True:
            out = tflearn.fully_connected(
                net, self.a_dim, activation='tanh', weights_init=w_init)
        else:
            out = tflearn.fully_connected(
                net, self.a_dim, activation='softmax', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        # TODO
        powerful_critic = True
        if powerful_critic is True:
            net = tflearn.fully_connected(inputs, 400)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases
            t1 = tflearn.fully_connected(net, 300)
            t2 = tflearn.fully_connected(action, 300)

            net = tflearn.activation(
                tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        else:
            net = tflearn.fully_connected(inputs, 400)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases
            t1 = tflearn.fully_connected(net, 600)
            t2 = tflearn.fully_connected(action, 600)

            net = tflearn.activation(
                tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

            # # Add the action tensor in the 2nd hidden layer
            # # Use two temp layers to get the corresponding weights and biases
            # t1 = tflearn.fully_connected(inputs, 300)
            # t2 = tflearn.fully_connected(action, 300)
            #
            # net = tflearn.activation(
            #     tf.matmul(inputs, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

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

def train(sess, env, args, actor, critic, actor_noise):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Epsilon parameter
    epsilon = args['epsilon_max']

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Time step
    time_step = 0.

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        ep_steps = 0

        while True:

            if args['render_env_flag']:
                env.render()

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            # TODO: different exploration strategy
            a = []
            action = []
            exploration_strategy = args['exploration_strategy']
            if exploration_strategy == 'action_noise':
                a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
                # Convert continuous action into discrete action
                if args['continuous_act_space_flag'] is True:
                    action = a[0]
                else:
                    action = np.argmax(a[0])
            elif exploration_strategy == 'epsilon_greedy':
                if np.random.rand() < epsilon:
                    if args['continuous_act_space_flag'] is True:
                        a = np.reshape(env.action_space.sample(), (1, actor.a_dim))
                    else:
                        a = np.random.uniform(0, 1, (1, actor.a_dim))
                else:
                    a = actor.predict(np.reshape(s, (1, actor.s_dim)))
                # Convert continuous action into discrete action
                if args['continuous_act_space_flag'] is True:
                    action = a[0]
                else:
                    action = np.argmax(a[0])
            else:
                print('Please choose a proper exploration strategy!')

            s2, r, terminal, info = env.step(action)

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Reduce epsilon
            time_step += 1.
            epsilon = args['epsilon_min'] + (args['epsilon_max'] - args['epsilon_min']) * np.exp(-args['epsilon_decay'] * time_step)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                if args['double_ddpg_flag']:
                    # Calculate targets: Double DDPG
                    target_q = critic.predict_target(
                            s2_batch, actor.predict(s2_batch))
                else:
                    # Calculate targets
                    target_q = critic.predict_target(
                            s2_batch, actor.predict_target(s2_batch))
                
                

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])
                
                # Update target networks
                if args['target_hard_copy_flag']:
                    if ep_steps % args['target_hard_copy_interval'] == 0:
                        actor.update_target_network()
                        critic.update_target_network()
                else:
                    actor.update_target_network()
                    critic.update_target_network()

            s = s2
            ep_reward += r
            ep_steps += 1

            # if terminal or reach maximum length
            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float((ep_steps + 1))
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                ep_stats = '| Episode: {0} | Steps: {1} | Reward: {2:.4f} | Qmax: {3:.4f}'.format(i,
                                                                                             (ep_steps + 1),
                                                                                             ep_reward,
                                                                                             (ep_ave_max_q / float(ep_steps+1)))
                print(ep_stats)
                logging.info(ep_stats)
                break

def main(args):

    with tf.Session() as sess:

        env = gym.make(args['env'])

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        # Set action_dim for continuous and discrete action space
        if args['continuous_act_space_flag'] is True:
            action_dim = env.action_space.shape[0]
            action_bound = env.action_space.high
            # Ensure action bound is symmetric
            assert (env.action_space.high == -env.action_space.low).all()
        else:
            action_dim = env.action_space.n
            # If discrete action, actor uses Softmax and action_bound is always 1
            action_bound = 1

        # Use hardcopy way to update target NNs.
        if args['target_hard_copy_flag'] is True:
            args['tau'] = 1.0

        actor = ActorNetwork(sess, args['continuous_act_space_flag'],
                             state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        # Record videos
        # Use the gym env Monitor wrapper
        if args['use_gym_monitor_flag']:
            monitor_dir = os.path.join(args['summary_dir'], 'gym_monitor')
            env = wrappers.Monitor(env, monitor_dir,
                                   resume=True,
                                   video_callable=lambda count: count % args['record_video_every'] == 0)

        train(sess, env, args, actor, critic, actor_noise)

        if args['use_gym_monitor_flag']:
            env.monitor.close()
        else:
            env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', type=float, default=0.0001, help='actor network learning rate')
    parser.add_argument('--critic-lr', type=float, default=0.001, help='critic network learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for critic updates')
    parser.add_argument('--tau', type=float, default=0.001, help='soft target update parameter')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='max size of the replay buffer')
    parser.add_argument('--minibatch-size', type=int, default=64, help='size of minibatch for minibatch-SGD')
    parser.add_argument("--continuous-act-space-flag", action="store_true", help='act on continuous action space')

    parser.add_argument("--exploration-strategy", type=str, choices=["action_noise", "epsilon_greedy"],
                        default='epsilon_greedy', help='action_noise or epsilon_greedy')
    parser.add_argument("--epsilon-max", type=float, default=1.0, help='maximum of epsilon')
    parser.add_argument("--epsilon-min", type=float, default=.01, help='minimum of epsilon')
    parser.add_argument("--epsilon-decay", type=float, default=.001, help='epsilon decay')

    # train parameters
    parser.add_argument('--double-ddpg-flag', action="store_true", help='True, if run double-ddpg-flag. Otherwise, False.')
    parser.add_argument('--target-hard-copy-flag', action="store_true", help='Target network update method: hard copy')
    parser.add_argument('--target-hard-copy-interval', type=int, default=200, help='Target network update hard copy interval')

    # run parameters
    # HalfCheetah-v2, Ant-v2, InvertedPendulum-v2, Pendulum-v0
    parser.add_argument('--env', type=str, default='HalfCheetah-v2', help='choose the gym env- tested on {Pendulum-v0}')
    parser.add_argument('--random-seed', type=int, default=1234, help='random seed for repeatability')
    parser.add_argument('--max-episodes', type=int, default=50000, help='max num of episodes to do while training')
    # parser.add_argument("--max-episode-len", type=int, default=1000, help='max length of 1 episode')
    parser.add_argument("--render-env-flag", action="store_true", help='render environment')
    parser.add_argument("--use-gym-monitor-flag", action="store_true", help='record gym results')
    parser.add_argument("--record-video-every", type=int, default=1, help='record video every xx episodes')
    parser.add_argument("--monitor-dir", type=str, default='./results/gym_ddpg', help='directory for storing gym results')
    parser.add_argument("--summary-dir", type=str, default='./results/tf_ddpg/HalfCheetah-v2/ddpg_Tau_0.001_run1', help='directory for storing tensorboard info')


    parser.set_defaults(use_gym_monitor=False)

    # args = vars(parser.parse_args())
    # args = parser.parse_args()
    args = vars(parser.parse_args())

    pp.pprint(args)

    if not os.path.exists(args['summary_dir']):
        os.makedirs(args['summary_dir'])
    log_dir = os.path.join(args['summary_dir'], 'ddpg_running_log.log')
    logging.basicConfig(filename=log_dir, filemode='a', level=logging.INFO)
    for key in args.keys():
        logging.info('{0}: {1}'.format(key, args[key]))

    main(args)

    # python ddpg_discrete_action.py --env