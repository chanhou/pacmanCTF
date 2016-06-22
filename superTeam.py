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
import numpy as np

#################
# Team creation #
#################
def createTeam(index, isRed,
               first = 'OffensiveQAgent', second = 'DeffensiveQAgent', third = 'OffensiveQAgent', **args):
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
      reward = (state.getScore() - self.lastState.getScore())      
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
    self.carryLimit = 10
    self.PowerTimer = 3

    # # initialize weights
    if self.numTraining == 0:
      self.epsilon = 0.0    # no exploration
      self.alpha = 0.0      # no learning
      # if os.path.exists(self.filename):
      #   print 'loading weights...'
      #   with open(self.filename, "rb") as f:
      #     self.weights = pickle.load(f)
      self.weights = util.Counter({
        'ghost-distance': 3.763708484777899, 
        'successorScore': 26.91631145259291, 
        'distanceToFood': -3.552103735787748, 
        'bias': -61.42192130600966, 
        'back-home': 0.05505876072946386, 
        '#-of-ghosts-1-step-away': -0.8667431307889246, 
        'eats-food': 0.0})
      # self.weights = util.Counter({'closest-food': -5.054839131498016, 'successorScore': 129.42310359521304, '#-of-ghosts-1-step-away': -2.271172287391008, 'bias': -147.63840814884603, 'escape': 1.4102253271888339, 'eats-food': -3.6297797537243683})

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    ApproximateQAgent.final(self, state)
    print self.index, self.getWeights()

  def observationFunction(self, state):
    """
      This is where we ended up after our last action.
      The simulation should somehow ensure this is called
    """
    if not self.lastState is None:
      reward = (state.getScore() - self.lastState.getScore())
      
      # pseudo reward
      reward -= 1. # one time step cost

      self.observeTransition(self.lastState, self.lastAction, state, reward)

    return CaptureAgent.observationFunction(self, state)


  def getFeatures(self, state, action):
    # Design Reward for carrying dots, eaten by ghost, minus by time, 

    # extract the grid of food and wall locations and get the ghost locations
    food = self.getFood(state)
    walls = state.getWalls()
    myPrevState = state.getAgentState(self.index)
    myPrePos = myPrevState.getPosition()
    teammatePositions = [state.getAgentPosition(teammate)for teammate in self.getTeam(state)]

    # capsulePos = self.getCapsules(state) 
    capsulePos = state.getRedCapsules() if state.isOnRedTeam(self.index) else state.getBlueCapsules()
    # isHome = state.isRed(myPrePos)
    # enemy = self.getOpponents(state)
    otherTeam = state.getBlueTeamIndices() if state.isOnRedTeam(self.index) else state.getRedTeamIndices()

    features = util.Counter()
    features["bias"] = 1.0

    successor = self.getSuccessor(state, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)/60.
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    checkall = 0
    # if myPrevState.isPacman: # uncomment it for optimal action, use it for more interesting randomness result
      # check for the distance between ghost
      # unindent from {
    dis = []
    for index in otherTeam:
      otherAgentState = state.data.agentStates[index]
      if otherAgentState.scaredTimer > self.PowerTimer: # power capsule 
        checkall += 1
      if otherAgentState.isPacman: continue
      ghostPosition = otherAgentState.getPosition()
      if ghostPosition == None: continue
      if otherAgentState.scaredTimer <= self.PowerTimer:
        features["#-of-ghosts-1-step-away"] = int(myPos in Actions.getLegalNeighbors(ghostPosition, walls))
        dis += [float(self.getMazeDistance(ghostPosition, myPos))]
    if len(dis)!=0: features['ghost-distance'] = min(dis)
    # if len(dis)!=0: print features['ghost-distance'], dis
      # to here}

    # dynamically change when need to return to got the score
    if myPrevState.numCarrying==0:
      self.carryLimit = len(foodList)
    # dynamically change if we meet the ghost then return to got the score
    if (features["#-of-ghosts-1-step-away"] and myPrevState.numCarrying!=0) or (state.data.timeleft/1./state.getNumAgents()/2. < walls.width):
    # if (state.data.timeleft/1./state.getNumAgents()/2. < walls.width):
        self.carryLimit = myPrevState.numCarrying if myPrevState.numCarrying != 0 else 2

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      dis = sorted([self.getMazeDistance(myPos, food) for food in foodList])
      # minDistance = dis[0] if not features['ghost-distance'] else dis[random.randint(1,len(foodList)-1)]
      minDistance = dis[0]
    else:
      minDistance = 0
    features['distanceToFood'] = float(minDistance)

    # if len(checkall)==len(otherTeam): # power of capsule
    #   # bug of this function
    #   features["#-of-ghosts-1-step-away"] = 0
    #   features['ghost-distance'] = 0
    #   # self.carryLimit = self.carryLimit
    #   # features['distanceToFood'] = -features['distanceToFood']
    # else:

    # condition of return to home
    back_home = False
    if checkall!=len(otherTeam):
      if myPrevState.numCarrying >= self.carryLimit:
        back_home = True
      else:
        if features['ghost-distance'] and (features['ghost-distance'] <= 3) and myPrevState.isPacman:
          back_home = True
        elif features['ghost-distance'] and not myPrevState.isPacman:
          back_home = True

    if len(foodList)==0:
      back_home = True

    if back_home:
      # features['back-home'] =  - 1.*self.getMazeDistance(capsulePos[0], myPos) / (walls.width * 5.)
      features['back-home'] =  - 1.*self.getMazeDistance(self.start, myPos) / (walls.width * 5.)
      features['distanceToFood'] = 0.
      # features['successorScore'] = 0.


    # if action == Directions.STOP: features['stop'] = 1
    # rev = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
    # if action == rev: features['reverse'] = 1

    # features['ghost-distance'] /= -6.
    features['ghost-distance'] = 0
    features['distanceToFood'] /= 1. *(walls.width * walls.height)
    features.divideAll(10.0)

    return features

  def getFeatures2(self, state, action):
    # Design Reward for carrying dots, eaten by ghost, minus by time, 

    # extract the grid of food and wall locations and get the ghost locations
    food = self.getFood(state)
    walls = state.getWalls()
    myPrevState = state.getAgentState(self.index)
    myPrePos = myPrevState.getPosition()
    # teammatePositions = [state.getAgentPosition(teammate)for teammate in self.getTeam(state)]

    # capsulePos = self.getCapsules(state) 
    # isHome = state.isRed(myPrePos)
    # enemy = self.getOpponents(state)
    otherTeam = state.getBlueTeamIndices() if state.isOnRedTeam(self.index) else state.getRedTeamIndices()

    features = util.Counter()
    features["bias"] = 1.0

    successor = self.getSuccessor(state, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)/21.
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    checkall = 0
    # if myPrevState.isPacman: # uncomment it for optimal action, use it for more interesting randomness result
      # check for the distance between ghost
      # unindent from {
    dis = []
    for index in otherTeam:
      otherAgentState = state.data.agentStates[index]
      if otherAgentState.scaredTimer > self.PowerTimer: # power capsule 
        checkall += 1
      if otherAgentState.isPacman: continue
      ghostPosition = otherAgentState.getPosition()
      if ghostPosition == None: continue
      features["#-of-ghosts-1-step-away"] = 0 if not features["#-of-ghosts-1-step-away"] else features["#-of-ghosts-1-step-away"]
      if otherAgentState.scaredTimer <= self.PowerTimer:
        features["#-of-ghosts-1-step-away"] += int(myPos in Actions.getLegalNeighbors(ghostPosition, walls))
        dis += [float(self.getMazeDistance(ghostPosition, myPos))]
    # if len(dis)!=0: features['ghost-distance'] = min(dis)
      # to here}
    # print food, myPos
    dist = closestFood((int(myPos[0]),int(myPos[1])), food, walls)
    if dist is not None:
      features["closest-food"] = float(dist) / (walls.width * walls.height)

    # legalAC = Actions.getLegalNeighbors((int(myPos[0]),int(myPos[1])), walls)
    legalAc = Actions.getPossibleActions(myState.configuration, walls)

    if not features["#-of-ghosts-1-step-away"] and food[int(myPos[0])][int(myPos[1])]:
      features["eats-food"] = 1.0
    if features["#-of-ghosts-1-step-away"] and len(legalAc)<=2:
      # rev = Directions.REVERSE[myPrevState.configuration.direction]
      for act in legalAc:
        if act == Directions.STOP: continue
        rev = Actions.reverseDirection(act)
        if action!=rev:
          features['escape'] = -1
          # features["closest-food"] = 0
          # features['successorScore'] = 0
    features.divideAll(10.0)

    return features


class DeffensiveQAgent(ApproximateQAgent):
  """
     Deffensive ApproximateQLearningAgent

     Only have to overwrite
     __init__() and final() functions.
  """
  def __init__(self, index, timeForComputing = .1, extractor='DeffenseExtractor', **args):
    ApproximateQAgent.__init__(self, index, timeForComputing, **args)
    # self.featExtractor = util.lookup(extractor, globals())()
    self.filename = "deffensive.train"
    # self.weights = util.Counter({'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'bias': 0, 'distanceToMiddleD': -10, 'scaredDistance': -10})
    # # initialize weights

    if self.numTraining == 0:
      self.epsilon = 0.0    # no exploration
      self.alpha = 0.0      # no learning
      self.weights = util.Counter({
        'numInvaders': 16.470010138965772, 
        'distanceToMiddleD': 5.532150686258845, 
        'successorScore': 171.19528245826652, 
        'scaredDistance': 0.0855915228745781, 
        'invaderDistance': -0.2610076844304779, 
        'bias': -197.97963869186557})
      # if os.path.exists(self.filename):
      #   print 'loading weights...'
          # self.epsilon = 0.0    # no exploration
          # self.alpha = 0.0      # no learning
      #   with open(self.filename, "rb") as f:
      #     self.weights = pickle.load(f)

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    ApproximateQAgent.final(self, state)
    print self.index, self.getWeights()

  def observationFunction(self, state):
    """
      This is where we ended up after our last action.
      The simulation should somehow ensure this is called
    """
    if not self.lastState is None:
      reward = (state.getScore() - self.lastState.getScore())
      
      # pseudo reward
      reward -= 1. # one time step cost

      self.observeTransition(self.lastState, self.lastAction, state, reward)

    return CaptureAgent.observationFunction(self, state)


  def getFeatures(self, state, action):

    food = self.getFoodYouAreDefending(state)
    walls = state.getWalls()
    myPrevState = state.getAgentState(self.index)
    myPosition = myPrevState.getPosition()
    # get middle

    #midPosition=[(walls.width/2-1, i) for i in range(1,walls.height-1)]
    #for i in midPosition:
    #  if state.hasWall(i[0], i[1]):
    #    midPosition.remove(i)
    midPosition = [(15, 1), (15, 2), (15, 4), (15, 5), (15, 7), (15, 8), (15, 11), (15, 12), (15, 13), (15, 14)]
    midPositionT = [(i[0]+1, 15-i[1]) for i in midPosition]

    # teammatePositions = [state.getAgentPosition(teammate)for teammate in self.getTeam(state)]

    successor = self.getSuccessor(state, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    features = util.Counter()

    features["bias"] = 1.0

    '''features["bias"] = 1.0 
    features['onDefense'] = 1.0
    if myState.isPacman: features['onDefense'] = 0.0'''
    # successor = self.getSuccessor(state, action)
    foodList = self.getFoodYouAreDefending(state).asList()


    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = -len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      # tempL = []
      # for i in midPosition:
      #   tempa = -1
      #   temp = 1000000
      #   for a in invaders:
      #     '''try:
      #       if self.getMazeDistance(i, a.getPosition()) < self.getMazeDistance(myPos, a.getPosition()):
      #         tempL.append(self.getMazeDistance(myPos, i)-self.getMazeDistance(i, a.getPosition()))
      #       else:
      #         tempL.append(0)
      #     except:
      #       tempL.append(0)
      #       continue'''
      #     if temp > self.getMazeDistance(i, a.getPosition()):
      #       temp = self.getMazeDistance(i, a.getPosition())
      #       tempa = i
      # tempL.append(self.getMazeDistance(tempa, myPos))
      #features['distanceToMiddleDiffer'] = max(tempL) * 1.0 /(walls.width * walls.height)
      # features['distanceToMiddleD'] = min(tempL) * 1.0 /(walls.width * walls.height)
      # tempL = [self.getMazeDistance(myPos, i) for i in midPosition]
      # features['distanceToMiddleD'] = min(tempL) * -1.0 /(walls.width * walls.height)
      features['invaderDistance'] = min(dists) * 1.0 /(walls.width * walls.height)
      try:
        capsulePos = self.getCapsulesYouAreDefending(state)
        if myState.scaredTimer > 0:
          features['scaredDistance'] = (min(dists) - myState.scaredTimer) * 1.0 /(walls.width * walls.height)
        else:
          features['scaredDistance'] = 0
      except:
        features['scaredDistance'] = 0
    else:
    # features['scaredDistance'] = 0
      tempL = sorted([self.getMazeDistance(myPos, i) for i in midPosition])
      # if myPos[0] < 13:
      features['distanceToMiddleD'] = max(tempL) * -1.0 /(walls.width * walls.height)
    # features['back-home'] =  - 1.*self.getMazeDistance(self.start, myPos) / (walls.width * 5.)

    # if action == Directions.STOP: features['stop'] = 1
    # rev = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
    # if action == rev: features['reverse'] = -1

    '''enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)/60.
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = float(minDistance)'''

    features.divideAll(10.0)

    return features