#!/usr/bin/python

#
# Original implementation:
#
# https://gist.github.com/BrandonLMorris/dc75086b844d65c51ab92b956494ecbd
#

import warnings
warnings.filterwarnings("ignore")
#
# Found GPU0 GeForce GTX 760 which is of cuda capability 3.0.
# PyTorch no longer supports this GPU because it is too old.
#
# This is packaging warning, and should be ignored when
# pytorch is compiled from source.
#

import torch
import torch.nn.functional as F
import itertools
import numpy as np

from torch import nn
from random import sample, randint, random
from time import time, sleep
from absl import app, flags           # command line parameters handling
from tqdm import trange               # terminal progress bar
from skimage.transform import resize  # image preprocessing

from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
from vizdoom.vizdoom import ViZDoomUnexpectedExitException

FLAGS = flags.FLAGS

frame_repeat = 12
resolution = (30, 45)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_config_file_path = "basic.cfg"


def preprocess(img):
	return torch.from_numpy(resize(img, resolution, mode="constant", anti_aliasing=False).astype(np.float32))


def game_state(game):
	return preprocess(game.get_state().screen_buffer)


class ReplayMemory:
	def __init__(self, capacity):
		channels = 1
		state_shape = (capacity, channels, resolution[0], resolution[1])
		self.s1 = torch.zeros(state_shape, dtype=torch.float32).to(device)
		self.s2 = torch.zeros(state_shape, dtype=torch.float32).to(device)
		self.a = torch.zeros(capacity, dtype=torch.long).to(device)
		self.r = torch.zeros(capacity, dtype=torch.float32).to(device)
		self.isterminal = torch.zeros(capacity, dtype=torch.float32).to(device)

		self.capacity = capacity
		self.size = 0
		self.pos = 0

	def add_transition(self, s1, action, s2, isterminal, reward):
		idx = self.pos
		self.s1[idx,0,:,:] = s1
		self.a[idx] = action

		if not isterminal:
			self.s2[idx,0,:,:] = s2

		self.isterminal[idx] = isterminal
		self.r[idx] = reward

		self.pos = (self.pos + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def get_sample(self, size):
		idx = sample(range(0, self.size), size)

		return (
			self.s1[idx],
			self.a[idx],
			self.s2[idx],
			self.isterminal[idx],
			self.r[idx],
		)


class QNet(nn.Module):
	def __init__(self, available_actions_count):
		super(QNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3) # 8x9x14
		self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2) # 8x4x6 = 192
		self.fc1 = nn.Linear(192, 128)
		self.fc2 = nn.Linear(128, available_actions_count)

		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.SGD(self.parameters(), FLAGS.learning_rate)
		self.memory = ReplayMemory(capacity=FLAGS.replay_memory)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(-1, 192)
		x = F.relu(self.fc1(x))
		return self.fc2(x)

	def get_best_action(self, state):
		q = self(state)
		_, index = torch.max(q, 1)
		return index

	def train_step(self, s1, target_q):
		output = self(s1)
		loss = self.criterion(output, target_q)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss

	def learn_from_memory(self):
		if self.memory.size < FLAGS.batch_size: return
		s1, a, s2, isterminal, r = self.memory.get_sample(FLAGS.batch_size)
		q = self(s2).detach()
		q2, _ = torch.max(q, dim=1)
		target_q = self(s1).detach()
		idxs = (torch.arange(target_q.shape[0]), a)
		target_q[idxs] = r + FLAGS.discount * (1-isterminal) * q2
		self.train_step(s1, target_q)


def find_eps(epoch):
	"""Balance exploration and exploitation as we keep learning"""
	start, end = 1.0, 0.1
	const_epochs, decay_epochs = .1*FLAGS.epochs, .6*FLAGS.epochs

	if epoch < const_epochs:
		return start
	elif epoch > decay_epochs:
		return end

	# Linear decay
	progress = (epoch-const_epochs)/(decay_epochs-const_epochs)

	return start - progress * (start - end)


def perform_learning_step(epoch, game, model, actions):
	s1 = game_state(game)

	if random() <= find_eps(epoch):
		# This is to prevent exploitation.
		a = torch.tensor(randint(0, len(actions) - 1)).long()
		#
		# tensor(4)
		#
	else:
		s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
		a = model.get_best_action(s1.to(device))

	reward = game.make_action(actions[a], frame_repeat)

	if game.is_episode_finished():
		isterminal, s2 = 1., None
	else:
		isterminal = 0.
		s2 = game_state(game)

	model.memory.add_transition(s1, a, s2, isterminal, reward)
	model.learn_from_memory()


def initialize_vizdoom(config):
	game = DoomGame()

	game.load_config(FLAGS.config)
	game.set_doom_game_path("vizdoom/doom2.wad")
	game.set_doom_scenario_path("vizdoom/scenarios/basic.wad")
	game.set_window_visible(False)
	game.set_sound_enabled(False)
	game.set_mode(Mode.PLAYER)
	game.set_screen_format(ScreenFormat.GRAY8)
	game.set_screen_resolution(ScreenResolution.RES_640X480)

	game.init()

	return game


def train(game, model, actions):
	time_start = time()
	print("Saving the network weigths to: ", FLAGS.save_path)

	for epoch in range(FLAGS.epochs):
		print("Epoch {}".format(epoch + 1))
		episodes_finished = 0
		scores = np.array([])
		game.new_episode()
		for learning_step in trange(FLAGS.iters):
			perform_learning_step(epoch, game, model, actions)
			if game.is_episode_finished():
				score = game.get_total_reward()
				scores = np.append(scores, score)
				game.new_episode()
				episodes_finished += 1
		print("Completed {} episodes".format(episodes_finished))
		print("Mean: {:.1f} +/- {:.1f}".format(scores.mean(), scores.std()))
		print("Testing...")
		test(FLAGS.test_episodes, game, model, actions)
		torch.save(model, FLAGS.save_path)

	print("Total elapsed time: {:.2f} minutes".format((time()-time_start)))


def test(iters, game, model, actions):
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
			state = state.reshape([1, 1, resolution[0], resolution[1]])
			#
			# tensor([[[[0.2158, 0.1373, 0.1373,  ..., 0.1373, 0.2294, 0.2458],
			#          [0.3098, 0.3582, 0.2863,  ..., 0.3468, 0.2863, 0.3020],
			#          [0.1804, 0.1373, 0.1831,  ..., 0.1373, 0.1408, 0.2693],
			#          ...,
			#          [0.3272, 0.3408, 0.3691,  ..., 0.3431, 0.3176, 0.3529],
			#          [0.3529, 0.3529, 0.3529,  ..., 0.3843, 0.3529, 0.3020],
			#          [0.4000, 0.4000, 0.4000,  ..., 0.4000, 0.3918, 0.3856]]]])
			#
			a_idx = model.get_best_action(state.to(device))
			game.make_action(actions[a_idx], frame_repeat)
		r = game.get_total_reward()
		scores = np.append(scores, r)
	print("Results: mean: {:.1f} +/- {:.1f}".format(scores.mean(), scores.std()))


def watch_episodes(game, model, actions):
	game.set_window_visible(True)
	game.set_mode(Mode.ASYNC_PLAYER)
	game.init()
	for episode in range(FLAGS.watch_episodes):
		try:
			episode_name = "Episode-{}".format(episode)
			print("============ {} ==============".format(episode_name))
			game.new_episode(episode_name)
			while not game.is_episode_finished():
				state = game_state(game)
				state = state.reshape([1, 1, resolution[0], resolution[1]])
				a_idx = model.get_best_action(state.to(device))
				game.set_action(actions[a_idx])
				for _ in range(frame_repeat):
					game.advance_action()
			sleep(1.0)
			score = game.get_total_reward()
			print("Total score: {}".format(score))
		except ViZDoomUnexpectedExitException:
			return


def main(_):
	game = initialize_vizdoom(FLAGS.config)

	n = game.get_available_buttons_size()

	actions = [list(a) for a in itertools.product([0, 1], repeat=n)]

	if FLAGS.load_model:
		print("Loading model from: {}".format(FLAGS.save_path))
		model = torch.load(FLAGS.save_path).to(device)
	else:
		model = QNet(len(actions)).to(device)

	if not FLAGS.skip_training:
		print("Starting the training!")
		train(game, model, actions)

	game.close()
	watch_episodes(game, model, actions)


if __name__ == '__main__':
	flags.DEFINE_integer('batch_size', 64, 'Batch size')
	flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate')
	flags.DEFINE_float('discount', 0.99, 'Discount factor')
	flags.DEFINE_integer('replay_memory', 10000, 'Replay memory capacity')
	flags.DEFINE_integer('epochs', 10, 'Number of epochs')
	flags.DEFINE_integer('iters', 2000, 'Iterations per epoch')
	flags.DEFINE_integer('watch_episodes', 10, 'Trained episodes to watch')
	flags.DEFINE_integer('test_episodes', 100, 'Episodes to test with')
	flags.DEFINE_string('config', default_config_file_path, 'Path to the config file')
	flags.DEFINE_boolean('skip_training', False, 'Set to skip training')
	flags.DEFINE_boolean('load_model', False, 'Load the model from disk')
	flags.DEFINE_string('save_path', 'model-doom.pth', 'Path to save/load the model')

	app.run(main)
