from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict
from gym.spaces import Space
from gym.spaces.utils import flatdim
import numpy as np


class Agent(ABC):
    """Base class for Q-Learning agent

    **ONLY CHANGE THE BODY OF THE act() FUNCTION**
    """

    def __init__(
        self,
        action_space: Space,
        obs_space: Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of the Q-Learning agent
        namely the epsilon, learning rate and discount rate.

        :param action_space (int): action space of the environment
        :param obs_space (int): observation space of the environment
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """

        self.action_space = action_space
        self.obs_space = obs_space
        self.n_acts = flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs: int) -> int:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (int): received observation representing the current environmental state
        :return (int): index of selected action
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q2")
        ### RETURN AN ACTION HERE ###

        act_vals = [self.q_table[(obs,act)] for act in range(self.n_acts)]
        
        max_val = max(act_vals)
        
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val==max_val]
        
        if random.random() < self.epsilon:
            
            return random.randint(0, self.n_acts-1)
        
        else:
        
            return random.choice(max_acts)
        
        #return -1

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class QLearningAgent(Agent):
    """
    Agent using the Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(
        self, obs: int, action: int, reward: float, n_obs: int, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (int): received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (int): received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q2")
        
        target_value = reward + self.gamma*(1-done)*self.q_table[(n_obs, action)]
        
        self.q_table[(obs, action)] += self.alpha*(target_value - self.q_table[(obs, action)])
        
        return self.q_table[(obs, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q2")
        
        max_deduct, decay = 0.9999, 0.05 #0.05
        
        #self.epsilon = 1.0- (min(1.0, timestep/(decay*50)))
        
        self.epsilon = 1.0- (min(1.0, timestep/(decay*max_timestep)))*max_deduct
        
        #print(max_timestep)
        #if self.epsilon > 0:
            #print(self.epsilon)


class MonteCarloAgent(Agent):
    """
    Agent using the Monte-Carlo algorithm for training

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """
#*max_timestep)))*max_deduct#*1500
    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = {}

    def learn(
        self, obses: List[int], actions: List[int], rewards: List[float]
    ) -> Dict:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obses (List(int)): list of received observations representing environmental states
            of trajectory (in the order they were encountered)
        :param actions (List[int]): list of indices of applied actions in trajectory (in the
            order they were applied)
        :param rewards (List[float]): list of received rewards during trajectory (in the order
            they were received)
        :return (Dict): A dictionary containing the updated Q-value of all the updated state-action pairs
            indexed by the state action pair.
        """
        updated_values = {}
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q2")
        """
        
        for idx, state in enumerate(obses):
            
            if (state, actions[idx]) not in updated_values:
                
                r = np.mean(rewards[idx:])
                
                updated_values[(state, actions[idx])] = r
            
                #Update q-table:
                self.q_table[(state, actions[idx])] += r    
                
        """
       
        G = 0
       
        #obses = list(reversed(obses))
        #actions = list(reversed(actions))
        #rewards = list(reversed(rewards))
        
        # SUM of rewards for every state action pairs:
        #sa_sums = dict()
        
        #self.sa_counts = dict()
       
        for idx in range(len(obses)-1,-1,-1):
            
            state = obses[idx]
            action = actions[idx]
           
            G = self.gamma*G + rewards[idx]
            
            temp = []
            
            for j in range(0,idx):                
                temp.append((obses[j],actions[j]))
                        
            
            if (state, action) not in self.sa_counts.keys():
                self.sa_counts[(state, action)] = 0
                
           
            if (state, action) not in temp:

                # Update Q value for this state-action pair:
                self.q_table[(state, action)] = (self.q_table[(state, action)]*self.sa_counts[(state, action)] + 
                            G)/(self.sa_counts[(state, action)]+1)
               
                updated_values[(state, action)] = self.q_table[(state, action)]
                
                self.sa_counts[(state, action)] += 1
        #print(self.epsilon)
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q2")
        
        #max_deduct, decay = 0.9999, 0.05 #0.05
        max_deduct, decay = 0.55, 0.05 #0.05
        
        #self.epsilon = 1.0- (min(1.0, timestep/(decay*50)))
        
        self.epsilon = 1.0 - (min(1.0, timestep/(decay*max_timestep)))*max_deduct
