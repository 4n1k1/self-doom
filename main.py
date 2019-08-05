#!/usr/bin/env python

import os

from random import sample, randint, random
from time import time, sleep
import numpy as np

from absl import app, flags           # command line parameters handling
from absl.flags import FLAGS
from tqdm import trange               # terminal progress bar
from skimage.transform import resize  # image preprocessing

import vizdoom
from vizdoom.vizdoom import ViZDoomUnexpectedExitException

import torch
from torch import nn


FRAME_REPEAT = 12
RESOLUTION = (30, 45)
DEFAULT_CONFIG_FILE_PATH = "basic.cfg"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess(img):
    return torch.from_numpy(
        resize(img, RESOLUTION, mode="constant", anti_aliasing=False).astype(np.float32)
    )


def game_state(game):
    return preprocess(game.get_state().screen_buffer)


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, RESOLUTION[0], RESOLUTION[1])
        self.state_0_buffer = torch.zeros(state_shape, dtype=torch.float32).to(DEVICE)
        self.state_1_buffer = torch.zeros(state_shape, dtype=torch.float32).to(DEVICE)
        self.actions = torch.zeros(capacity, dtype=torch.long).to(DEVICE)
        self.rewards = torch.zeros(capacity, dtype=torch.float32).to(DEVICE)
        self.is_terminal = torch.zeros(capacity, dtype=torch.float32).to(DEVICE)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, state_0, action, state_1, is_terminal, reward):
        idx = self.pos
        self.state_0_buffer[idx, 0, :, :] = state_0
        self.actions[idx] = action

        if not is_terminal:
            self.state_1_buffer[idx, 0, :, :] = state_1

        self.is_terminal[idx] = is_terminal
        self.rewards[idx] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, size):
        idx = sample(range(0, self.size), size)

        return (
            self.state_0_buffer[idx],
            self.actions[idx],
            self.state_1_buffer[idx],
            self.is_terminal[idx],
            self.rewards[idx],
        )


class QNet(nn.Module):
    def __init__(self, available_actions_count):
        super(QNet, self).__init__()
        self.convolution_0 = nn.Conv2d(1, 8, kernel_size=6, stride=3)  # 8x9x14
        self.convolution_1 = nn.Conv2d(8, 8, kernel_size=3, stride=2)  # 8x4x6 = 192
        self.full_connections_0 = nn.Linear(192, 128)
        self.full_connections_1 = nn.Linear(128, available_actions_count)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), FLAGS.learning_rate)
        self.memory = ReplayMemory(capacity=FLAGS.memory_capacity)

    def forward(self, state):
        state = nn.functional.relu(self.convolution_0(state))
        state = nn.functional.relu(self.convolution_1(state))
        state = state.view(-1, 192)
        state = nn.functional.relu(self.full_connections_0(state))
        return self.full_connections_1(state)

    def get_best_action(self, state):
        output = self(state)
        _, index = torch.max(output, 1)
        return index

    def learn_from_memory(self):
        if self.memory.size < FLAGS.batch_size:
            return

        states_t0, t0_to_t1_actions, states_t1, is_terminal, rewards = self.memory.get_sample(FLAGS.batch_size)

        t0_to_t1_action_probabilities = self(states_t0)
        picked_actions_t0 = t0_to_t1_action_probabilities.gather(1, t0_to_t1_actions.unsqueeze(1)).squeeze(1)

        t1_to_t2_action_probabilities = self(states_t1).detach()
        most_possible_actions_t1, _ = torch.max(t1_to_t2_action_probabilities, dim=1)

        target_actions = FLAGS.discount * most_possible_actions_t1 * (1 - is_terminal) + rewards

        loss = self.criterion(picked_actions_t0, target_actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def find_eps(epoch):
    """Balance exploration and exploitation as we keep learning"""
    start, end = 1.0, 0.1
    const_epochs, decay_epochs = .1*FLAGS.epochs, .6*FLAGS.epochs

    if epoch < const_epochs:
        return start

    if epoch > decay_epochs:
        return end

    # Linear decay
    progress = (epoch-const_epochs)/(decay_epochs-const_epochs)

    return start - progress * (start - end)


def perform_learning_step(epoch, game, model, actions):
    state_0 = game_state(game)

    if random() <= find_eps(epoch):
        # This is to prevent exploitation.
        action = torch.tensor(randint(0, len(actions) - 1)).long()
    else:
        state_0 = state_0.reshape([1, 1, RESOLUTION[0], RESOLUTION[1]])
        action = model.get_best_action(state_0.to(DEVICE))

    reward = game.make_action(actions[action], FRAME_REPEAT)

    if game.is_episode_finished():
        is_terminal, state_1 = 1., None
    else:
        is_terminal = 0.
        state_1 = game_state(game)

    model.memory.add_transition(state_0, action, state_1, is_terminal, reward)
    model.learn_from_memory()


def initialize_vizdoom():
    vizdoom_home = os.path.split(vizdoom.__file__)[0]

    game = vizdoom.DoomGame()

    game.load_config(FLAGS.config)
    game.set_doom_game_path(f"{vizdoom_home}/doom2.wad")
    game.set_doom_scenario_path(f"{vizdoom_home}/scenarios/basic.wad")
    game.set_window_visible(False)
    game.set_sound_enabled(FLAGS.skip_training)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)

    game.init()

    return game


def train(game, model, actions):
    time_start = time()
    print("Saving the network weights to: ", FLAGS.save_path)

    for epoch in range(FLAGS.epochs):
        print(f"Epoch {epoch + 1}")
        episodes_finished = 0
        scores = np.array([])
        game.new_episode()
        for _ in trange(FLAGS.iterations):
            perform_learning_step(epoch, game, model, actions)
            if game.is_episode_finished():
                score = game.get_total_reward()
                scores = np.append(scores, score)
                game.new_episode()
                episodes_finished += 1
        print(f"Completed {episodes_finished} episodes")
        print(f"Mean: {scores.mean():.1f} +/- {scores.std():.1f}")
        print("Testing...")
        test(game, model, actions)
        torch.save(model, FLAGS.save_path)

    print(f"Total elapsed time: {time()-time_start:.2f} minutes")


def test(game, model, actions):
    scores = np.array([])
    for _ in trange(FLAGS.test_episodes, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = game_state(game)
            #
            # tensor([[0.3007, 0.3098, 0.3569,  ..., 0.3098, 0.3412, 0.3569],
            #        [0.1373, 0.2627, 0.1373,  ..., 0.1373, 0.3072, 0.1373],
            #        [0.2771, 0.2941, 0.3098,  ..., 0.2941, 0.2889, 0.3098],
            #        ...,
            #        [0.3529, 0.3529, 0.3185,  ..., 0.3529, 0.3529, 0.3529],
            #        [0.3020, 0.3529, 0.3431,  ..., 0.3185, 0.3193, 0.3412],
            #        [0.3667, 0.3431, 0.3333,  ..., 0.3667, 0.3529, 0.3431]])
            #
            state = state.reshape([1, 1, RESOLUTION[0], RESOLUTION[1]])
            #
            # tensor([[[[0.2158, 0.1373, 0.1373,  ..., 0.1373, 0.2294, 0.2458],
            #          [0.3098, 0.3582, 0.2863,  ..., 0.3468, 0.2863, 0.3020],
            #          [0.1804, 0.1373, 0.1831,  ..., 0.1373, 0.1408, 0.2693],
            #          ...,
            #          [0.3272, 0.3408, 0.3691,  ..., 0.3431, 0.3176, 0.3529],
            #          [0.3529, 0.3529, 0.3529,  ..., 0.3843, 0.3529, 0.3020],
            #          [0.4000, 0.4000, 0.4000,  ..., 0.4000, 0.3918, 0.3856]]]])
            #
            a_idx = model.get_best_action(state.to(DEVICE))
            game.make_action(actions[a_idx], FRAME_REPEAT)
        reward = game.get_total_reward()
        scores = np.append(scores, reward)
    print(f"Results: mean: {scores.mean():.1f} +/- {scores.std():.1f}")


def watch_episodes(game, model, actions):
    game.set_window_visible(True)
    game.set_mode(vizdoom.Mode.ASYNC_PLAYER)
    game.init()
    for episode in range(FLAGS.watch_episodes):
        try:
            episode_name = f"Episode-{episode}"
            print(f"============ {episode_name} ==============")
            game.new_episode(episode_name)
            while not game.is_episode_finished():
                state = game_state(game)
                state = state.reshape([1, 1, RESOLUTION[0], RESOLUTION[1]])
                a_idx = model.get_best_action(state.to(DEVICE))
                game.set_action(actions[a_idx])
                for _ in range(FRAME_REPEAT):
                    game.advance_action()
            sleep(1.0)
            score = game.get_total_reward()
            print(f"Total score: {score}")
        except ViZDoomUnexpectedExitException:
            return


def main(_):
    game = initialize_vizdoom()

    actions = []

    for idx, _ in enumerate(game.get_available_buttons()):
        action = [0, 0, 0]
        action[idx] = 1
        actions.append(action)

    if FLAGS.load_model:
        print(f"Loading model from: {FLAGS.save_path}")
        model = torch.load(FLAGS.save_path).to(DEVICE)
    else:
        model = QNet(len(actions)).to(DEVICE)

    if not FLAGS.skip_training:
        print("Starting the training!")
        train(game, model, actions)

    game.close()

    watch_episodes(game, model, actions)


if __name__ == '__main__':
    flags.DEFINE_integer('batch_size', 64, 'Batch size')
    flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate')
    flags.DEFINE_float('discount', 0.99, 'Discount factor')
    flags.DEFINE_integer('memory_capacity', 10000, 'Replay memory capacity')
    flags.DEFINE_integer('epochs', 10, 'Number of epochs')
    flags.DEFINE_integer('iterations', 2000, 'Iterations per epoch')
    flags.DEFINE_integer('watch_episodes', 10, 'Trained episodes to watch')
    flags.DEFINE_integer('test_episodes', 100, 'Episodes to test with')
    flags.DEFINE_string('config', DEFAULT_CONFIG_FILE_PATH, 'Path to the config file')
    flags.DEFINE_boolean('skip_training', False, 'Set to skip training')
    flags.DEFINE_boolean('load_model', False, 'Load the model from disk')
    flags.DEFINE_string('save_path', 'model-doom.pth', 'Path to save/load the model')

    app.run(main)
