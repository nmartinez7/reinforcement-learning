from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict

from gym.spaces import Space
from gym.spaces.utils import flatdim


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self) -> List[int]:
        """Chooses an action for all agents for stateless task

        :return (List[int]): index of selected action for each agent
        """
        ...

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


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """
        actions = []
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q5")
        
        #print(self.q_tables[0])
        #print(self.n_acts)
        
        for i in range(self.num_agents):
            
            act_vals = [self.q_tables[i][act] for act in range(self.n_acts[i])]
            
            #print("act vals: {}".format(act_vals))
        
            max_val = max(act_vals)
        
            max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val==max_val]
        
            
            if random.random() < self.epsilon:
            
                action = random.randint(0, self.n_acts[i]-1)
        
            else:
        
                action = random.choice(max_acts)
                
            actions.append(action)    
        
        #print(actions)
        
        return actions

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current actions of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q5") 
        
        #print("actions: {}".format(actions))
        #print("rewards: {}".format(rewards))
        #print("dones: {}".format(dones))
        
        for i in range(self.num_agents):
            
            act_vals = [self.q_tables[i][act] for act in range(self.n_acts[i])]
        
            max_val = max(act_vals)
            
            target = rewards[i] + self.gamma*(1 - dones[i])*max_val - self.q_tables[i][actions[i]]
                       
            value = self.q_tables[i][actions[i]] + self.learning_rate*target
            
            updated_values.append(value)
            
            self.q_tables[i][actions[i]] = value
        
        #print(self.epsilon)
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q5")
        
        max_deduct, decay = 0.98, 0.05 #0.05
        
        #self.epsilon = 1.0- (min(1.0, timestep/(decay*50)))
        
        self.epsilon = 1.0 - (min(1.0, timestep/(decay*max_timestep)))*max_deduct


class JointActionLearning(MultiAgent):
    """
    Agents using the Joint Action Learning algorithm with Opponent Modelling

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping joint actions ACTs
            to respective Q-values for all agents
        :attr models (List[DefaultDict]): each agent holding model of other agent
            mapping other agent actions to their counts

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: 0) for _ in range(self.num_agents)] 

    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """
        joint_action = []
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q5")
        
        for i in range(self.num_agents):
            
            #print("model: {}".format(self.models[i]))
            #print("q_tables: {}".format(self.q_tables[i]))
            
            #act_vals = [self.models[i][act] for act in range(self.n_acts[i])]
            
            act_vals = []
            
            ###################################################################
            # Calculatr EV:            
            for j in range(self.n_acts[i]):
                
                ev_a = 0
                c = 0
                
                for k in range(self.num_agents):                  
                    
                    if i == k:
                        
                        continue
                    
                    else:
                        for m in range(self.n_acts[i]):
                        
                        #print("j value:{}".format(j))
                        
                            c += self.models[k][j]
                            #ev_a += self.models[k][j]*self.q_tables[i][4*j]
                            ev_a += self.models[k][j]*self.q_tables[i][(j,m)]
                        
                        ev_a = ev_a/max(1,c)
                
                
                        act_vals.append(ev_a)
                        
            #print("act_vals: {}".format(act_vals))
            #print("act vals: {}".format(act_vals))
            ###################################################################
            max_val = max(act_vals)
        
            max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val==max_val]
        
            
            if random.random() < self.epsilon:
            
                action = random.randint(0, self.n_acts[i]-1)
        
            else:
                
                action = random.choice(max_acts)
                
            joint_action.append(action)          
            
        return joint_action
    

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q5")
        
        """
        print("actions: {}".format(actions))
        print("rewards: {}".format(rewards))
        print("dones: {}".format(dones))
        print("models: {}".format(self.models[0]))
        """

        for i in range(self.num_agents):
            
            a = 0
            b = 1
            
            
            ########### Agent 0:
            if i == 0:
                self.models[b][actions[b]] +=1
            elif i == 1:
                self.models[a][actions[a]] +=1
            
            # Get max EV:
                        # Get max EV:
            ###############################################################
            act_vals = []
            
            ###################################################################
            # Calculatr EV:            
            for j in range(self.n_acts[0]):
                
                ev_a = 0
                c = 0
                
                for k in range(self.num_agents):                  
                    
                    if i == k:
                        
                        continue
                    
                    else:
                        
                        #print("j value:{}".format(j))
                        
                        for m in range(self.n_acts[0]):
                        
                            c += self.models[k][j]
                            ev_a += self.models[k][j]*self.q_tables[i][(j,m)]
                        
                ev_a = ev_a/max(1,c)
                #print(f"j value {j}")
                
                
                act_vals.append(ev_a)
                #print(act_vals)
            EV = max(act_vals)         
    
            ###################################################################
            
            # Do update for agent 0:
            if i==0:
        
                target = rewards[i] + self.gamma*EV*(1.0-float(dones[i])) - self.q_tables[i][(actions[a],actions[b])]
        
                value = self.q_tables[i][(actions[a],actions[b])] + self.learning_rate*target
        
                updated_values.append(value)
            
                self.q_tables[i][(actions[a],actions[b])] = value   
        
        
            # Do update for agent 1 (i.e. reverse order of actions when accessing Q)
            elif i==1:
                
                target = rewards[i] + self.gamma*EV*(1.0-float(dones[i])) - self.q_tables[i][(actions[b],actions[a])]
        
                value = self.q_tables[i][(actions[b],actions[a])] + self.learning_rate*target
        
                updated_values.append(value)
                
                #updated_values.append(value)
            
                self.q_tables[i][(actions[b],actions[a])] = value   
        
        #### Agent 1:
        #self.models[0][actions[1]] +=1
        
        #self.q_tables[1][(actions[1],actions[0])] = value   
        
        
        # Agent 1:
       

        return updated_values
    

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q5")
        
        max_deduct, decay = 0.9999, 0.1 #0.05
        
        #self.epsilon = 1.0- (min(1.0, timestep/(decay*50)))
        
        #self.epsilon = 1.0 - (min(1.0, timestep/(decay*max_timestep)))*max_deduct
        
        #self.learning_rate = 1.0 - (min(1.0, timestep/(decay*max_timestep)))*max_deduct