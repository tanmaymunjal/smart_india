# make all necessary import statements


from module import *
import tensorflow as tf
import tensorflow_lattice as tfl
from keras import layers
import tf_agents as tfa
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
import matplotlib.pyplot as plt
import reverb

# <---------------we define neural net parameters and models here----------------------------->
# Channel and Special attention classes imported from modules.py
# no sound in simulation, so sound processing functionality omitted

# creating sequential model of network

model = tf.keras.models.Sequential()
# tensorflow layer for simulated camera data
image_layer = tf.keras.models.Sequential([
    tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None),
    ChannelAttention(128, 8),
    SpatialAttention(7),
    layers.GlobalAveragePooling2D(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, 3, padding='same', activation='elu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='elu'),
    ChannelAttention(128, 8),
    SpatialAttention(7),
])

# tensorflow layer for simulated radar data
radar_layer = tf.keras.models.Sequential([
    tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None),
    ChannelAttention(128, 8),
    SpatialAttention(7),
    layers.GlobalAveragePooling2D(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, 3, padding='same', activation='elu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='elu'),
    ChannelAttention(128, 8),
    SpatialAttention(7),
])

# tensorflow layer for simulated lidar data
lidar_layer = tf.keras.models.Sequential([
    tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None),
    ChannelAttention(128, 8),
    SpatialAttention(7),
    layers.GlobalAveragePooling2D(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, 3, padding='same', activation='elu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='elu'),
    ChannelAttention(128, 8),
    SpatialAttention(7),
])

# final fusion layer in which outputs of all preceding three layers will be sent

fusion_layer = tf.keras.models.Sequential([
    tf.keras.layers.dense(100, activation='relu'),
    tf.keras.layers.dense(50, activation='relu'),
    tf.keras.layers.dense(25, activation='relu'),
    tf.keras.layers.dense(10, activation=None),
])

# <--pre-training all the models before training agent to allow for high-efficiency training of model in simulation---->
# this step will not be done in federated learning module as it is to be only done once

# compiling the models

radar_layer.compile(
    optimizer=tf.keras.optimizers.adam_v2,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=tf.keras.mectrics.Accuracy(),
)

lidar_layer.compile(
    optimizer=tf.keras.optimizers.adam_v2,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=tf.keras.mectrics.Accuracy(),
)

"""
Statements to be executed after importing radar and lidar data from internet, will most probably 
require cloud or gpu support(will probably prefer cloud due to ease of setup, personal gpu to setup and program from
start to finish will take a lot of time and effort to start and run)
radar_layer.fit(x_train_radar, y_train_radar, batch_size=64, epochs=10)
lidar_layer.fit(x_train_lidar,y_train_lidar,batch_size=64, epochs=10)
"""

# defining a parallel layer in which three neural networks will be added in parallel

main_layer = tfl.layers.ParallelCombination()
main_layer.append(image_layer)
main_layer.append(radar_layer)
main_layer.append(lidar_layer)

# the tensorflow model design is completed
model.add(main_layer)
model.add(fusion_layer)
# <---------------we define tensorflow agent here----------------------------------->

# defining agent hyper parameters
env_name = "Mathworks simulation"
num_iterations = 5000
collect_episodes_per_iteration = 5
replay_buffer_capacity = 2000

learning_rate = 1e-3
log_interval = 25
num_eval_episodes = 10
eval_interval = 50

# setting two environments(one for training and one for evaluation)

train_py_env = tfa.environments.suite_gym.load(env_name)
eval_py_env = tfa.environments.suite_gym.load(env_name)

# converting the environment to tensor form to speedup traning and evaluation
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# creating the ego agent distribution network

input_tensor_spec = tf.TensorSpec(shape=[10], dtype=tf.int64, name="Input data")
output_tensor_spec = tf.TensorSpec(shape=[10], dtype=tf.int64, name="Policy decision")
actor_net=actor_distribution_network.ActorDistributionNetwork(
      input_tensor_spec,
      output_tensor_spec,
      preprocessing_layers=main_layer,
      fc_layer_params=(200, 100),
      name='The king of all ego actors')

# initialising some agent training hyper parameters

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

#building the training agent

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()

# setting initial agent policies
eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy


#  computing average return for agent policies for agent training
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for i in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    average_return = total_return / num_episodes
    return average_return


# program to store data from simulation and use during training

# WARNING: THIS PORTION OF CODE WILL ONLY WORK IN LINUS, IT WILL NOT WORK IN WINDOWS OR MACOS
# IT WILL SHOW COULDN'T FIND REFERENCE ERROR, PLEASE COMMENT THIS PORTION OUT IF YOU ARE
# NOT USING LINUS OR A LINUX BASED OS OR ARE ON MOBILE OR TABLET


table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
    tf_agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddEpisodeObserver(
    replay_buffer.py_client,
    table_name,
    replay_buffer_capacity
)


# program to store collected data(observation,action,rewards etc not whole simulation) in drive
def collect_episode(environment, policy, num_episodes):
    driver = py_driver.PyDriver(
        environment,
        py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=True),
        [rb_observer],
        max_episodes=num_episodes)
    initial_time_step = environment.reset()
    driver.run(initial_time_step)


# training the agent

# Optimize by wrapping some code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(
        train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    train_loss = tf_agent.train(experience=trajectories)

    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

# program to print efficiency of agent over time

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=5000)
