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
               first = 'OffensiveQAgent', second = 'OffensiveQAgent', third = 'OffensiveQAgent', **args):
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
    return [eval(first)(index[0], **args)]
  elif len(index) == 2:
    return [eval(first)(index[0], **args), eval(second)(index[1], **args)]
  elif len(index) == 3:
    return [eval(first)(index[0], **args), eval(second)(index[1], **args), eval(third)(index[2], **args)]

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
  def __init__(self, index, timeForComputing = .1, numTraining=0, epsilon=0.5, alpha=0.5, gamma=1, **args):
    """
    actionFn: Function which takes a state and returns the list of legal actions - REMOVED

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
    self.startEpisode()
    if self.episodesSoFar == 0:
      print 'Beginning %d episodes of Training' % (self.numTraining)

  def observationFunction(self, state):
    """
      This is where we ended up after our last action.
      The simulation should somehow ensure this is called
    """
    if not self.lastState is None:
      reward = (state.getScore() - self.lastState.getScore())*100
      
      # pseudo reward
      reward -= 1. # one time step cost
      agentState = state.data.agentStates[self.index]
      preagentState = self.lastState.data.agentStates[self.index]
      otherTeam = state.getBlueTeamIndices()
      if agentState.numCarrying > preagentState.numCarrying:
        reward += 20
      if agentState.numReturned > preagentState.numReturned:
        reward += 100

      if agentState.isPacman:
        for index in otherTeam:
          otherAgentState = state.data.agentStates[index]
          if otherAgentState.isPacman: continue
          ghostPosition = otherAgentState.getPosition()
          if ghostPosition == None: continue
          if distanceCalculator.manhattanDistance( ghostPosition, agentState.getPosition() ) <= 0.5:
            # award points to the other team for killing Pacmen
            if otherAgentState.scaredTimer <= 0:
              reward -= 10.
            else:
              reward += 1.
      else: # pacman is ghost
        for index in otherTeam:
          otherAgentState = state.data.agentStates[index]
          if not otherAgentState.isPacman: continue
          pacPos = otherAgentState.getPosition()
          if pacPos == None: continue
          if distanceCalculator.manhattanDistance( pacPos, agentState.getPosition() ) <= 0.5:
            #award points to the other team for killing Pacmen
            if agentState.scaredTimer <= 0:
              reward += 10.
            else:
              reward -= 10
      self.observeTransition(self.lastState, self.lastAction, state, reward)

    return CaptureAgent.observationFunction(self, state)
    # return state

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we have never seen a state
      or the Q node value otherwise
    """
    return self.qValues[(state, action)]

  def computeValueFromQValues(self, gameState):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    actions = state.getLegalActions(self.index)
    
    if len(actions) == 0:
      return 0.0
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.getQValue(gameState, a) for a in actions]
    # print 'eval time for agent %:wd: %.4f' % (self.index, time.time() - start)

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
    if len(bestActions) == 0:
      return None
    return random.choice(bestActions)

  def doAction(self,state,action):
    """
        Called by inherited class when
        an action is taken in a state
    """
    self.lastState = state
    self.lastAction = action

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
    action = None
    if util.flipCoin(self.epsilon):
      action = random.choice(actions)
    else:
      action = self.computeActionFromQValues(gameState)

    self.doAction(gameState,action) # from Q learning agent
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

  def observeTransition(self, state,action,nextState,deltaReward):
    """
        Called by environment to inform agent that a transition has
        been observed. This will result in a call to self.update
        on the same arguments

        NOTE: Do *not* override or call this function
    """
    self.episodeRewards += deltaReward
    self.update(state,action,nextState,deltaReward)

  def startEpisode(self):
    """
      Called by environment when new episode is starting
    """
    self.lastState = None
    self.lastAction = None
    self.episodeRewards = 0.0

  def stopEpisode(self):
    """
      Called by environment when episode is done
    """
    if self.episodesSoFar < self.numTraining:
      self.accumTrainRewards += self.episodeRewards
    else:
      self.accumTestRewards += self.episodeRewards
    self.episodesSoFar += 1
    if self.episodesSoFar >= self.numTraining:
      # Take off the training wheels
      self.epsilon = 0.0    # no exploration
      self.alpha = 0.0      # no learning

  def isInTraining(self):
    return self.episodesSoFar < self.numTraining

  def isInTesting(self):
    return not self.isInTraining()

  def final(self, state):
    """
      Called by Pacman game at the terminal state
    """
    CaptureAgent.final(self, state)
    deltaReward = state.getScore() - self.lastState.getScore()
    self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
    self.stopEpisode()

    # Make sure we have this var
    if not 'episodeStartTime' in self.__dict__:
        self.episodeStartTime = time.time()
    if not 'lastWindowAccumRewards' in self.__dict__:
        self.lastWindowAccumRewards = 0.0
    self.lastWindowAccumRewards += state.getScore()

    NUM_EPS_UPDATE = 100
    if self.episodesSoFar % NUM_EPS_UPDATE == 0:
        print 'Reinforcement Learning Status:'
        windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
        if self.episodesSoFar <= self.numTraining:
            trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
            print '\tCompleted %d out of %d training episodes' % (
                   self.episodesSoFar,self.numTraining)
            print '\tAverage Rewards over all training: %.2f' % (
                    trainAvg)
        else:
            testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
            print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
            print '\tAverage Rewards over testing: %.2f' % testAvg
        print '\tAverage Rewards for last %d episodes: %.2f'  % (
                NUM_EPS_UPDATE,windowAvg)
        print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
        self.lastWindowAccumRewards = 0.0
        self.episodeStartTime = time.time()

    if self.episodesSoFar == self.numTraining:
        msg = 'Training Done (turning off epsilon and alpha)'
        print '%s\n%s' % (msg,'-' * len(msg))
        print self.getWeights()


class ApproximateQAgent(QLearningAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update. All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, index, timeForComputing = .1, extractor='IdentityExtractor', **args):
    QLearningAgent.__init__(self, index, timeForComputing, **args)
    # self.featExtractor = util.lookup(extractor, globals())()
    self.weights = util.Counter()

  def getWeights(self):
    return self.weights

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    weights = self.getWeights()
    # features = self.featExtractor.getFeatures(state, action)
    features = self.getFeatures(state, action)
    return weights * features

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    actions = nextState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.getQValue(nextState, a) for a in actions]
    # print 'eval time for agent %wd: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)

    weights = self.getWeights()
    # features = self.featExtractor.getFeatures(state, action)
    features = self.getFeatures(state, action)
    for feature in features:
      difference = (reward + self.discount * maxValue) - self.getQValue(state, action)
      self.weights[feature] = weights[feature] + self.alpha * difference * features[feature]

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    QLearningAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      with open(self.filename, 'wb') as f:
        pickle.dump(self.weights, f)
      pass

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def getFeatures(self, state, action):
    util.raiseNotDefined()
    # return None

class OffensiveQAgent(ApproximateQAgent):
  """
     Offensive ApproximateQLearningAgent

     Only have to overwrite
     __init__() and final() functions.
  """
  def __init__(self, index, timeForComputing = .1, extractor='OffenseExtractor', **args):
    ApproximateQAgent.__init__(self, index, timeForComputing, **args)
    # self.featExtractor = util.lookup(extractor, globals())()
    self.filename = "offensive.train"
    # if os.path.exists(self.filename):
    #   print 'loading weights...'
        # self.epsilon = 0.0    # no exploration
        # self.alpha = 0.0      # no learning
    #   with open(self.filename, "rb") as f:
    #     self.weights = pickle.load(f)
    ''' 
    'ghost-distance': 0.05328764440886632, 
    'successorScore': 21.234303877546264, 
    'distanceToFood': -3.589551647328648, 
    'bias': -61.68600087062215, 
    'back-home': 0.1537940016251592, 
    '#-of-ghosts-1-step-away': -0.6687040033467684
    
    Best result
    2 {'ghost-distance': 3.763708484777899, 'successorScore': 26.91631145259291, 'distanceToFood': -3.552103735787748, 'bias': -61.42192130600966, 'back-home': 0.05505876072946386, '#-of-ghosts-1-step-away': -0.8667431307889246, 'eats-food': 0.0}

    '''
    # initialize weights
    self.numTraining = 0
    self.epsilon = 0.0    # no exploration
    self.alpha = 0.0      # no learning
    # the best!
    self.weights = util.Counter({'ghost-distance': 3.763708484777899, 'successorScore': 26.91631145259291, 'distanceToFood': -3.552103735787748, 'bias': -61.42192130600966, 'back-home': 0.05505876072946386, '#-of-ghosts-1-step-away': -0.8667431307889246, 'eats-food': 0.0})

    # # second best !
    # self.weights["ghost-distance"] = 0.05328764440886632
    # self.weights["bias"] = -61.68600087062215
    # self.weights['successorScore']= 21.234303877546264
    # self.weights['distanceToFood'] = -3.589551647328648
    # self.weights["#-of-ghosts-1-step-away"] = -0.6687040033467684
    # self.weights['back-home'] = 0.1537940016251592    

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    ApproximateQAgent.final(self, state)
    print self.index, self.getWeights()

  def getFeatures(self, state, action):
    # Design Reward for carrying dots, eaten by ghost, minus by time, 

    # extract the grid of food and wall locations and get the ghost locations
    food = self.getFood(state)
    walls = state.getWalls()
    myPrevState = state.getAgentState(self.index)
    myPosition = myPrevState.getPosition()
    teammatePositions = [state.getAgentPosition(teammate)
            for teammate in self.getTeam(state)]

    capsulePos = self.getCapsules(state) 
    isHome = state.isRed(myPosition)

    # self.getMazeDistance((state.data.layout.width/2., myPosition[1]), myPosition)
    # disFromHome = distanceCalculator.manhattanDistance((state.data.layout.width/2., myPosition[1]), myPosition)

    enemy = self.getOpponents(state)
    otherTeam = state.getBlueTeamIndices()

    features = util.Counter()
    features["bias"] = 1.0

    successor = self.getSuccessor(state, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)/60.
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = float(minDistance) / (walls.width * walls.height)
      
    # myState.isPacman, numCarrying, numReturned, scaredTimer
    # self.start, 
    # get other team scaredTimer

      if myPrevState.isPacman or myState.isPacman:
        dis = []
        for index in otherTeam:
          otherAgentState = state.data.agentStates[index]
          # if otherAgentState.scaredTimer > 0:
          #   continue
          if otherAgentState.isPacman: continue
          ghostPosition = otherAgentState.getPosition()
          if ghostPosition == None: continue
          if otherAgentState.scaredTimer <= 0:
            features["#-of-ghosts-1-step-away"] = int(myPos in Actions.getLegalNeighbors(ghostPosition, walls))
            dis += [float(self.getMazeDistance(ghostPosition, myPos))/6.]
            # if distanceCalculator.manhattanDistance( ghostPosition, myState.getPosition() ) <= 0.5:
        if len(dis)!=0: features['ghost-distance'] = -min(dis)

      if myPrevState.numCarrying >=2:
        features['back-home'] = -1.*self.getMazeDistance(self.start,myPos) / (walls.width * 10.)
        # features['run-home'] = -1.
        features['distanceToFood'] = 0.
        features['successorScore'] = 0.
      else:
        if not features['ghost-distance']:
            features["eats-food"] = 0.0
        else:
          if features['ghost-distance'] < -2 and features['distanceToFood']<=1:
            features["eats-food"] = 0.0
          else:
            features['back-home'] = -1.*self.getMazeDistance(self.start,myPos) / walls.width * 1.
            # features['run-home'] = -1.
            features['distanceToFood'] = 0.
            features['successorScore'] = 0.

      # if len(capsulePos)!=0:
      #   features['dis-from-capsules'] = float(min([ self.getMazeDistance(myPos, dis) for dis in capsulePos]))/ (walls.width * walls.height)

      # if not features["#-of-ghosts-1-step-away"] and self.getFood(state)[int(myPos[0])][int(myPos[1])]:

      # if not features["#-of-ghosts-1-step-away"] and myPrevState.numCarrying <2 and self.getFood(state)[int(myPos[0])][int(myPos[1])]:
      #   features["eats-food"] = 1.0

      features.divideAll(10.0)

      # need to normalize
      # if not isHome:
      #   features['dis-from-home'] = -1.*float(disFromHome) #/ (walls.width * walls.height)
      # else:
      #   features['escape-home'] = -1*1.*float(disFromHome) # / (walls.width * walls.height)


      # if (next_x, next_y) in capsulePos:
      #     feature['power'] = 1.0

      # if (isHome):
      #     feature['is-home'] = 1.0

      # print walls.width, walls.height, myPosition, capsulePos, [ self.getMazeDistance(myPosition, dis) for dis in capsulePos], teammatePositions

      # if there is no danger of ghosts then add the food feature
      # if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:


      # dist = closestFood((next_x, next_y), food, walls)
      # if dist is not None:
      #     # make the distance a number less than one otherwise the update
      #     # will diverge wildly
      #     features["closest-food"] = float(dist) / (walls.width * walls.height)

    return features
