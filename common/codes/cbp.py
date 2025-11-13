from torch import optim
#from lop.algos.gnt import GnT
#from lop.utils.AdamGnT import AdamGnT
from gnt import GnT
from AdamGnT import AdamGnT
import torch.nn.functional as F
import torch

class ContinualBackprop(object):
    """
    The Continual Backprop algorithm, used in https://arxiv.org/abs/2108.06325v3
    """
    def __init__(
            self,
            net,
            step_size=0.001,
            loss='mse',
            opt='sgd',
            beta=0.9,
            beta_2=0.999,
            replacement_rate=0.001,
            decay_rate=0.9,
            device='cpu',
            maturity_threshold=100,
            util_type='contribution',
            init='kaiming',
            accumulate=False,
            momentum=0,
            outgoing_random=False,
            weight_decay=0
    ):
        self.net = net

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        elif opt == 'adam':
            self.opt = AdamGnT(self.net.parameters(), lr=step_size, betas=(beta, beta_2), weight_decay=weight_decay)
        
        self.loss_name = loss
        # define the loss function
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[loss]

        # a placeholder
        self.previous_features = None

        # define the generate-and-test object for the given network
        self.gnt = None
        self.gnt = GnT(
            net=self.net.layers, #neural net layers
            hidden_activation=self.net.act_type, #activation
            opt=self.opt,                     #optimization
            replacement_rate=replacement_rate,  #rate at which neurons are reinitialized
            decay_rate=decay_rate,
            maturity_threshold=maturity_threshold,   #when a neuron is reinitialized and has utility zero we dont want it to get reinitialized again in next time step
            util_type=util_type,
            device=device,
            loss_func=self.loss_func,
            init=init,             #kaiming
            accumulate=accumulate,  #boolean value
        )

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent and generate-&-test
        :param x: input
        :param target: desired output
        :return: loss
        """
        
        # do a forward pass and get the hidden activations
        output, features = self.net.predict(x=x) # output is the output of nn, features is output after the final hidden layer afer activation
        loss = self.loss_func(output, target)
        self.previous_features = features

        # do the backward pass and take a gradient step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # take a generate-and-test step
        self.opt.zero_grad()
        if type(self.gnt) is GnT:
            self.gnt.gen_and_test(features=self.previous_features)

        if self.loss_func == F.cross_entropy:
            return loss.detach(), output.detach()
  
            
        return loss.detach()
  

      
    def test(self, x, target ):
         
         if self.loss_name  =='mse':
             
             output, features = self.net.predict(x=x)
             
             loss = self.loss_func(output, target)      
             
             return loss.detach()
         
         else:
             
            output, features = self.net.predict(x=x) # output is the output of nn, features is output after the final hidden layer afer activation
            
            y_pred = output.argmax(dim = 1) 
            
            #was len(target.unique()) for previous exeriment. Change it according to requirement.
            accuracies =  100 * (y_pred == target ) .float().sum() / len(target) 
             
            return accuracies
        
    def calculate_hessian(self,x, target):
        
        #params = list(self.net.out_layer.fc.parameters() )
        
        params = list(self.net.layers[-1].parameters())
        
        output, features = self.net.predict(x=x)
        
        loss = self.loss_func(output, target)
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        grads_flat = torch.cat([g.reshape(-1) for g in grads])
        
        # Compute Hessian (final layer only)
        num_params = grads_flat.numel()
        
        H = torch.zeros((num_params, num_params))
        
        for i in range(num_params):
            second_grads = torch.autograd.grad(grads_flat[i], params, retain_graph=True)
            
            H[i] = torch.cat([g.reshape(-1) for g in second_grads]).detach()
        
        # Effective rank of Hessian
        #eigenvalues = torch.linalg.eigvalsh(H)
        
        try:
            epsilon = 1e-6
            H_stable = H + epsilon * torch.eye(H.shape[0])
            eigenvalues = torch.linalg.eigvalsh(H_stable)
        except:
            print("stop")
            print("stop")
            print("stop")
            
        eigenvalues = torch.clamp(eigenvalues, min=1e-12)  # Prevent log(0)
        
        p = eigenvalues / eigenvalues.sum()
        
        entropy = -torch.sum(p * torch.log(p))
        
        effective_rank = torch.exp(entropy)
        
        return effective_rank    
        
        
        