from abc import ABC, abstractmethod
from copy import deepcopy
import gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

from rl2022.exercise3.networks import FCNetwork
from rl2022.exercise3.replay import Transition


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see http://gym.openai.com/docs/#spaces for more information on Gym spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
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
    def update(self):
        ...


class DQN(Agent):
    """DQN agent

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        **kwargs,
    ):
        """The constructor of the DQN agent class

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
        )

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = 1
        # ######################################### #
        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q3")
        
              
        #self.epsilon = 1.0- (min(1.0, timestep/(decay*50)))
        max_deduct, decay = 0.999, 0.0001#0.05
        
        #self.epsilon = 1.0 - (min(1.0, timestep/(decay*max_timestep)))*max_deduct
        
        self.epsilon = max(0.001, 0.95*self.epsilon)
        
        #https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html#Define-some-hyperparameter

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q3")

        
        # Create random number for exploration:
        rn = np.random.randn() 
        
        if (explore == True):
              
            # Pick epsilon-greedy action:
            if rn < self.epsilon:
                action = self.action_space.sample()
                
            else:
                
                with torch.no_grad():
                    q_values = self.critics_net(Tensor(obs))
                action = torch.argmax(q_values).detach().numpy()
                
            
        elif (explore == False) :#or (rn > self.epsilon):
                #with torch.no_grad():
                q_values = self.critics_net(Tensor(obs)).detach()
                action = torch.argmax(q_values).detach().numpy()
                

        return int(action) 


    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network, update the target network at the given
        target update frequency, and return the Q-loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q3")
        # https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f
        #q_loss = 0.0
        q_loss = torch.tensor(0.0)
        q_loss.requires_grad=True
        
        N = len(batch[0])
        # "states", "actions", "next_states", "rewards", "done"
        states = batch[0]
        actions = batch[1]
        next_states = batch[2]
        rewards = batch[3]
        done = batch[4]
        
        #q_values_next = self.critics_target(next_states)
        #q_values = self.critics_net(states)
        """
        print("states :{}".format(states.size()))
        print("actions :{}".format(actions.size()))
        print("next_states :{}".format(next_states.size()))
        print("rewards :{}".format(rewards.size()))
        print("done :{}".format(done.size()))
        
        print("q_values :{}".format(q_values.size()))
        print("q_values_next :{}".format(q_values_next.size()))
        """
        
        q_values_next = self.critics_target(next_states).max(1)[0].unsqueeze(1).detach()
        
        #print("q_values_next: {}".format(q_values_next.size()))
        
        q_target = rewards + self.gamma*q_values_next*(1 - done)
        
        q_expected = self.critics_net(states).gather(1, actions.long())
        
        #q_loss = sum((target - q_expected)**2)
        
        q_loss = torch.nn.functional.mse_loss(q_target, q_expected)
    
        """
        for i in range(self.batch_size):
            #print(i)
            # Get the Q_values from the next state:
            #with torch.no_grad():
            #q_values_next = self.critics_target(next_states)
            #q_values = self.critics_net(state)
            #print("")
            #print("action:{}".format(int(actions[i].item())))
            q_loss_batch = (rewards[i] + self.gamma*(1 - done[i])*torch.max(q_values_next[i,:])- 
                            q_values[i,int(actions[i])])**2  
            
            if i == 0:
                q_loss = q_loss_batch#.clone()
            
            #q_loss += (rewards[i] + self.gamma*(1 - done[i])*torch.max(q_values_next[i,:]).item() - q_values[i,int(actions[i].item())])**2  
            else:
                
                q_loss += q_loss_batch
        """   
        # Divide by N since it is a batch average:        
        #q_loss = q_loss/N
        
        # Update value network:
        self.critics_optim.zero_grad()
        q_loss.backward()
        self.critics_optim.step()
        
        if self.update_counter%self.target_update_freq == 0:
            
            self.critics_target = deepcopy(self.critics_net)
            
            #self.critics_target.hard.update(self.critics_net)
            
         
        self.update_counter += 1
        return {"q_loss": q_loss.detach().numpy()}


class Reinforce(Agent):
    """Reinforce agent

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr policy (FCNetwork): fully connected network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for policy network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
        )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        # ###############################################
        self.saveables.update(
            {
                "policy": self.policy,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters 

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q3")
        
        max_deduct, decay = 0.9, 0.05 #0.05
        
        #self.epsilon = 1.0- (min(1.0, timestep/(decay*50)))
        
        #self.learning_rate = 1.0 - (min(1.0, timestep/(decay*max_timesteps)))*max_deduct
        
        self.learning_rate = self.learning_rate*np.exp(-timestep/max_timesteps) + 1e-5
        
        #print("learning_rate: {}".format(self.learning_rate))
        
        
        return

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q3")
        
        # Get probabibilities:
        with torch.no_grad():
            probs = self.policy(torch.FloatTensor(obs)).detach().numpy()
            
        # Pick a discrete action:
        action = np.random.choice(self.action_space.n, p = probs)
        #m = Categorical(self.policy(torch.FloatTensor(obs)))
        
        #action = m.sample().detach().numpy()
        #print(action)
        #print(self.learning_rate)
        return action

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
        ) -> Dict[str, float]:
        """Update function for policy gradients

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q3")
        p_loss = 0.0
        G = 0
        
        p_loss = torch.tensor(0.0)
        p_loss.requires_grad=True
        
        self.policy_optim.zero_grad()
        
        
        for idx in range(len(rewards)-1,-1,-1):
            
            G = rewards[idx] + self.gamma*G
            
            #print('idx: {}'.format(idx))
            #print('observations: {}'.format(observations[idx]))
            #print('rewards: {}'.format(rewards[idx]))
            
            #with torch.no_grad():
            probs = self.policy(torch.FloatTensor(observations[idx]))
            
            #print('probs: {}'.format(probs))
            #print('actions: {}'.format(actions[idx]))
            #print('selection: {}'.format(probs[actions[idx]]))
            p_loss = p_loss - G*torch.log(probs[actions[idx]])
            
        # Average loss (i.e. divide by T):
        p_loss = p_loss/len(rewards)
        
        #print('loss tensor: {}'.format(p_loss))
        #print(type(p_loss))
        
        # Perform gradient step:
        #self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()
        
        #print("learning rate: {}".format(self.learning_rate))
        
        return {"p_loss": p_loss}
