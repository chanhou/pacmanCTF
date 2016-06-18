# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from featureExtractors import *
import distanceCalculator
import random, time, util, sys
import pickle
import os.path
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################
def createTeam(index, isRed,
               first = 'OffensiveQAgent', second = 'ApproximateQAgent', third = 'ApproximateQAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  if len(index) == 1:
    return [eval(first)(index[0])]
  elif len(index) == 2:
    return [eval(first)(index[0]), eval(second)(index[1])]
  elif len(index) == 3:
    return [eval(first)(index[0]), eval(second)(index[1]), eval(third)(index[2])]

##########
# Agents #
##########

class QLearningAgent(CaptureAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - computeValueFromQValues
      - computeActionFromQValues
      - getQValue
      - getAction
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
  """
  def __init__(self, index, timeForComputing = .1, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
    """
    actionFn: Function which takes a state and returns the list of legal actions

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    CaptureAgent.__init__(self, index, timeForComputing)
    self.episodesSoFar = 0
    self.accumTrainRewards = 0.0
    self.accumTestRewards = 0.0
    self.numTraining = int(numTraining)
    self.epsilon = float(epsilon)
    self.alpha = float(alpha)
    self.discount = float(gamma)
    self.qValues = util.Counter()

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we have never seen a state
      or the Q node value otherwise
    """
    return self.qValues[(state, action)]

  def computeValueFromQValues(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    actions = state.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.getQValue(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    return max(values)

  def computeActionFromQValues(self, gameState):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.getQValue(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def chooseAction(self, gameState):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    actions = gameState.getLegalActions(self.index)
    if util.flipCoin(self.epsilon):
      action = random.choice(actions)
    else:
      action = self.computeActionFromQValues(gameState)
    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    self.qValues[(state, action)] = (1-self.alpha)*self.qValues[(state, action)] + self.alpha*(reward + self.discount*self.computeValueFromQValues(nextState))

  def getPolicy(self, state):
    return self.computeActionFromQValues(state)

  def getValue(self, state):
    return self.computeValueFromQValues(state)

class ApproximateQAgent(QLearningAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, index, timeForComputing = .1, extractor='IdentityExtractor', **args):
    QLearningAgent.__init__(self, index, timeForComputing, **args)
    self.featExtractor = util.lookup(extractor, globals())()
    self.weights = util.Counter()

  def getWeights(self):
    return self.weights

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    weights = self.getWeights()
    features = self.featExtractor.getFeatures(state, action)
    return weights * features

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    actions = nextState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)

    weights = self.getWeights()
    features = self.featExtractor.getFeatures(state, action)
    for feature in features:
      difference = (reward + self.discount * maxValue) - self.getQValue(state, action)
      self.weights[feature] = weights[feature] + self.alpha * difference * features[feature]

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    # QLearningAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      pass

class OffensiveQAgent(ApproximateQAgent):
  """
     Offensive ApproximateQLearningAgent

     Only have to overwrite
     __init__() and final() functions.
  """
  def __init__(self, index, timeForComputing = .1, extractor='OffenseExtractor', **args):
    ApproximateQAgent.__init__(self, index, timeForComputing, **args)
    self.filename = "offensive.train"
    self.featExtractor = util.lookup(extractor, globals())()
    if os.path.exists(self.filename):
      with open(self.filename, "rb") as f:
        self.weights = pickle.load(f)
    else:
      self.weights = util.Counter()

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    # QLearningAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining and self.numTraining > 0:
      with open(self.filename) as f:
        pickle.dump(self.weights, f)
      # you might want to print your weights here for debugging
      pass