# AI for Self Driving Car
# Writen with help of online material
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable # tensor and gradient

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):  #initialize network
        super(Network, self).__init__() #inheritence
        self.input_size = input_size
        self.nb_action = nb_action # output size
        self.fc1 = nn.Linear(input_size, 50) # make full connection between the neurons of the input layer and hidden layer
        self.fc2 = nn.Linear(50, nb_action)# connection betwwen hidden layer and output layer
    
    def forward(self, state): # activte neurons with rectifier activating function and get outpu q value
        x = F.relu(self.fc1(state)) # hidden neurons activation , relu: rectifier function
        q_values = self.fc2(x) # output neurons 
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size): #take random simples (batch size is the number of them) of the memory list
        samples = zip(*random.sample(self.memory, batch_size)) # reform the format
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #  put samples in pitorch variable

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = [] # sliding window of the last 100 reward
        self.model = Network(input_size, nb_action) # the actual neural network
        self.memory = ReplayMemory(200000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001) # tools to apply grdient descent, lr is learn rate, it is advisible to be low so that the AI can good learn
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # last state is a 5 dimentional vector 
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*75) # T=75
        #T ,tempature, the closer to 0 the less sure the nn will be of the action, the bigger t is the more sure the nn is
        # Variable is used to get the output of the neural network
        # state is a tensor and tensors are wrapped in a variable which includes a gradient
        # volatile=true to say that we dont need the gradianet of the input to the graph of all computation of the nn model to save memory and improve the  performance
        
        action = probs.multinomial()
        # take a random draw of the probability distribution of softmax
        return action.data[0,0]
        # get rid of the fake batch
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): # train nn with forward and back propagation
        
        #batch is like an array of many values to make the learning deep
        # the batch has a fake dimension at index 0 we remove it with squeeze
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # get output batch from the input batch
        # we use gather to get the output of only the action that was chosen, and not all the 3 outputs
        
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        #we need next output to calculate the target
                  
        
        target = self.gamma*next_outputs + batch_reward
        
        td_loss = F.smooth_l1_loss(outputs, target)
        # backpropagate with stochastic gradient descent
        
        
        self.optimizer.zero_grad()
        #reinitialize the optimizer at each loop the the gradient descent
        td_loss.backward(retain_variables = True)
        #back propagate, retain varialbes is to clear the memory
        self.optimizer.step()
        # update the weights
        
    def update(self, reward, new_signal):
        # update when the  ai discovers a new state
        # connect between ai and map
        
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #update new state
        
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        #upadte memroy
        #LongTensor is to create a tensor form a simple number
        
        action = self.select_action(new_state)
        # play the new action
        # we now need to learn from the last 100 events
        #make sure that we reached 100 events and take random samples from them to lear from
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #get random samples and learn from them
        
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        # update all parametetrs
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    # +1 to avoid that the demominatior is 0
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")