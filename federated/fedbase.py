import torch
import random
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class Server:
    def __init__(self, model_architecture, device='cpu'):
        self.model_architecture = model_architecture
        self.global_model = model_architecture().to(device)
        self.prev_update = None
        self.device = device  # Store device here
    
    # Helper: build model and set weights
    def get_weights(self):
        return [v.detach().cpu().numpy() for v in self.global_model.state_dict().values()]

    def set_weights(self, weights):
        state_dict = self.global_model.state_dict()
        for i, key in enumerate(state_dict):
            state_dict[key] = torch.tensor(weights[i], device=next(self.global_model.parameters()).device)
        self.global_model.load_state_dict(state_dict)

    def set_weights_others(self, weight_update):
        weights = [l + g for l, g in zip(self.get_weights(), weight_update)]
        self.set_weights(weights)

    def set_weights_fedmi(self, weight_update, beta):
        update = [beta * g + (1 - beta) * l for l, g in zip(self.get_weights(), weight_update)]
        weights = [l + g for l, g in zip(self.get_weights(), update)]
        self.set_weights(weights)

    def set_weights_autogm(self, model_weights):
        self.set_weights(model_weights)
    
    def test(self, X_test, Y_test):
        self.global_model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():  # No need to track gradients during evaluation
            inputs = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
            labels = torch.tensor(Y_test, dtype=torch.long).to(self.device)
            
            outputs = self.global_model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            # Compute loss
            cce = torch.nn.CrossEntropyLoss()
            loss = cce(outputs, labels).item()
            
            # Compute accuracy
            preds = torch.argmax(probs, dim=1)
            acc = accuracy_score(labels.cpu(), preds.cpu())
            
        return acc, loss
    
class Clients:
    def __init__(self, model_architecture, attack_type):
        self.model_architecture = model_architecture
        self.attack = attack_type
        self.num_clients = 0
        self.clients = dict()
        self.faulty_clients = []

    def create_clients(self, users, data, p_faulty):
        ''' return: a dictionary with keys clients' names and value as
                    data shards - tuple of datas and label lists.
            args:
                model: tensorflow model object
                data: a dictionary of training data for multiple users
                num_client: number of fedrated members (clients)
                initials: the clients'name prefix, e.g, clients_1
        '''
        num_clients = len(users)
        self.num_clients = num_clients
        self.faulty_clients = random.sample(range(1,num_clients+1),int(p_faulty*num_clients))

        for i in range(1, num_clients+1):
            model = self.model_architecture()
            image_array = np.array(data[i-1]['x']).reshape(-1, 1, 28, 28)
            label_array = np.array(data[i-1]['y'])
            if i in self.faulty_clients:
                self.clients[i] = {'model':model, 'data':image_array, 'labels':label_array, 'faulty':True}
            else:
                self.clients[i] = {'model':model, 'data':image_array, 'labels':label_array, 'faulty':False}

    def get_clients(self):
        return self.clients
    
    @staticmethod
    def sub(local_model, global_model):
        return [l - g for l, g in zip(local_model, global_model)]
    
    def generate_random_weights(self):
        model = self.model_architecture()
        state_dict = model.state_dict()
        random_state_dict = {}

        for name, param in state_dict.items():
            random_state_dict[name] = torch.randn_like(param)

        return random_state_dict
