"""
A bot that learns to play pong using policy gradients and
the openAI gym environment
"""

import numpy as np
import pickle
import gym

def preprocess(image):
    """ Preprocess a game snapshot
    Arguments:
        image - a snapshot of the current game state in form of an image
    Returns:
        the preprocessed image, ready to set as input for the model
    """
    # crop image
    image = image[35:195]

    # downsample by a factor of two
    image = image[::2,::2,0]

    # erase background colors and binarize the image
    image[image == 144] = 0
    image[image == 109] = 0
    image[image != 0] = 1

    return image.astype(np.float).ravel()

class PGLearner:

    """
    hyperparameters
    """

    LAYER1_SIZE = 200
    BATCH_SIZE = 10
    LEARNING_RATE = 1e-4

    # discount factor
    GAMMA = 0.99

    # rmsprop decay rate
    DECAY_RATE = 0.99

    # load existing model or not
    RESUME = True

    # number of pixels in the game
    TOTAL_PIXELS = 80*80


    def __init__(self, preprocess_fn, game='Pong-v0'):
        """ Initialize model """

        self.preprocess = preprocess_fn

        if self.RESUME:
            # load existing model if specified
            self.model = pickle.load(open('rl-pong-numpy-model.p','rb'))
        else:
            self.model = {}

            # xavier init technique for initialization
            self.model['W1'] = np.random.randn(self.LAYER1_SIZE, self.TOTAL_PIXELS) / np.sqrt(self.TOTAL_PIXELS) 
            self.model['W2'] = np.random.randn(self.LAYER1_SIZE) / np.sqrt(self.LAYER1_SIZE) 

        # create gradients and rmsprop storage for all the models weights
        self.grad_buffer = { k: np.zeros_like(v) for k, v in self.model.items() }
        self.rmsprop_cache = { k: np.zeros_like(v) for k, v in self.model.items() }

        # create instance of game
        self.env = gym.make(game)

    def sigmoid(self, x):
        """ The sigmoid activation function
        Arguments:
            x the input (z) to squash
        Returns:
            the output of the sigmoid
        """
        return 1.0 / (1.0 + np.exp(-x))



    def discount_rewards(self,r):
        """ Compute the discounted reward given the immediate rewards

        Arguments:
        r - a list of immediate rewards

        Returns:
        the discounted rewards
        """

        discounted_reward = np.zeros_like(r)
        ema = 0

        # iterate over all timesteps in the reward vector
        for t in reversed(range(0, r.size)):
            # reset running average
            if r[t] != 0: ema = 0

            # update running average
            ema = ema * self.GAMMA + r[t]

            # calucate discounted reward for current timestep
            discounted_reward[t] = ema

        return discounted_reward

    def policy_forward(self, x):
        """ define the forward propagation process

        Arguments:
        x - the input to the network

        Returns:
        A2 - the output of the network (probability of action up/down)
        Z1 - the output of layer 1 before relu, used later for backprop
        """

        # first layer forward prop: relu(Wx)
        Z1 = np.dot(self.model['W1'], x)
        A1 = Z1
        A1[A1 < 0] = 0 # activation using relu

        # second layer forward prop: sigmoid(Wx)
        Z2 = np.dot(self.model['W2'], A1)
        A2 = self.sigmoid(Z2)
        
        # return the output of the 2nd layer and Z1 as cache
        return A2, Z1

    def policy_backward(self, epx, eph, epdlogp):
        """ define the backward propagation process

        Arguments:
        eph - TODO
        epdlogp - 

        Returns:
        a dict of gradients for the trainable variables
        """

        # layer 2 gradients
        dW2 = np.dot(eph.T, epdlogp).ravel()
        
        # backprop through layer 2 to layer 1
        dA1 = np.outer(epdlogp, self.model['W2'])
        dZ1 = dA1
        dZ1[eph <= 0] = 0 # backpropagation trhough relu

        # layer 1 gradients
        dW1 = np.dot(dZ1.T, epx)

        # return gradients
        return { 'W1': dW1, 'W2': dW2 }

    def train(self):

        # initial observation by reseting the game
        observation = self.env.reset()

        prev_x = None

        # declare observations, hidden states, gradients, rewards
        xs, hs, dlogps, drs = [], [], [], []

        # reward storages
        running_reward = None
        total_reward = 0

        # episode counter
        episode_number = 0

        # training looop
        while True:

            # preprocess image
            current_x = self.preprocess(observation)

            # take the difference from the revious image with the current image
            x = current_x - prev_x if prev_x is not None else np.zeros(self.TOTAL_PIXELS)
            prev_x = current_x

            # forward propagation
            A2, Z1 = self.policy_forward(x)

            # perform a stochastic action with the network output into consideration
            # i.e. perform action 3 with the probability of the output of the network, 
            # otherwise action 2
            action = 2 if np.random.uniform() < A2 else 3

            # save network properties for caching to later use for backpropagation
            xs.append(x)
            hs.append(Z1)

            # create label from action
            y = 1 if action == 2 else 0
            dlogps.append(y - A2) # cost

            # take a step in the environment with the chosen action
            self.env.render()
            observation, reward, done, info = self.env.step(action)

            # add new reward
            total_reward += reward

            # cache reward
            drs.append(reward)

            if done:
                # episode finished
                episode_number += 1

                # stack together inputs, hidden states, action gradients and rewards for the episode
                epx = np.vstack(xs) # observation
                eph = np.vstack(hs) # hidden
                epdlogp = np.vstack(dlogps) # gradient
                epr = np.vstack(drs) # reward

                # reset caches
                xs, hs, dlogps, drs = [], [], [], [] 

                # compute discounted reward
                discounted_epr = self.discount_rewards(epr)
                # normalize
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                epdlogp *= discounted_epr

                # calculate gradients
                grad = self.policy_backward(epx, eph, epdlogp)

                # accumulate gradients to use later for rmsprop
                for k in self.model:
                    self.grad_buffer[k] += grad[k]

                # perform rmsprop parameter update only after a batch
                if episode_number % self.BATCH_SIZE == 0:
                    for k, v in self.model.items():
                        g = self.grad_buffer[k]
                        self.rmsprop_cache[k] = self.DECAY_RATE * self.rmsprop_cache[k] + (1 - self.DECAY_RATE) * g**2
                        self.model[k] += self.LEARNING_RATE * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)

                running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
                print('end of episode. total reward %f. running mean %f' % (total_reward, running_reward))

                # backup model every 100 episodes
                if episode_number % 100 == 0: pickle.dump(self.model, open('rl-pong-numpy-model2.p','wb'))

                # reset
                total_reward = 0
                observation = self.env.reset()
                prev_x = None

            # game finished
            if reward != 0:
                print('episode %d: game finished reward %f' % (episode_number, reward))

if __name__ == "__main__":
    pg = PGLearner(preprocess)
    pg.train()
