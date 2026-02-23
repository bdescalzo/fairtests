import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import FairMethod

device = "cuda" if torch.cuda.is_available() else "cpu"

# Modelo arbitrario a usar por defecto para MAML.
# TODO: Añadir que en el constructor se pueda personalizar este modelo (o pasar uno creado por el usuario)
class ModeloEnBruto(nn.Module):
    def __init__(self, tam_entrada): 
        super().__init__()
        self.fc1 = nn.Linear(tam_entrada, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, params=None):
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.linear(x, params['fc1.weight'], params['fc1.bias'])
            x = F.relu(x)
            x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
            x = F.relu(x)
            x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x.squeeze(-1)

class MetaLearning(FairMethod):
    def __init__(self, inner_lr=0.05, inner_steps=5, meta_epochs=40, meta_lr=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_epochs = meta_epochs
        self.meta_lr = meta_lr
        
        self.meta_model = None
        self.group_params = {} # To store adapted parameters for each group
        self.datos_cargados = False
        
        # We use BCEWithLogitsLoss because the model outputs raw logits
        self.loss_fn = nn.BCEWithLogitsLoss()

    def load_data(self, X_train, y_train, X_test):
        # movemos todos los datos a gpu si se puede
        # Convert to float explicitly to match model weights
        self.X_train = X_train.float().to(device)
        self.y_train = y_train.float().to(device)
        self.X_test = X_test.float().to(device)
        
        self.input_dim = self.X_train.shape[1]
        self.datos_cargados = True

    # Samplea batches de soporte/query para un grupo dado
    def _sample_task_batches(self, group_id, k_support=128, k_query=128):
        # Find indices where sensitive label matches group_id
        idxs = np.where(self.sensitive_train == group_id)[0]
        
        # Safety check for small groups
        current_k_support = min(len(idxs) // 2, k_support)
        current_k_query = min(len(idxs) // 2, k_query)
        if current_k_support < 1: 
            return None, None, None, None

        replace = len(idxs) < (current_k_support + current_k_query)
        chosen = np.random.choice(idxs, size=current_k_support + current_k_query, replace=replace)
        
        support_idx = chosen[:current_k_support]
        query_idx = chosen[current_k_support:]

        support_x = self.X_train[support_idx]
        support_y = self.y_train[support_idx]
        query_x = self.X_train[query_idx]
        query_y = self.y_train[query_idx]
        
        return support_x, support_y, query_x, query_y

    def fit(self, sensitive_labels, **kwargs):
        if not self.datos_cargados:
            print("No hay datos de entrenamiento cargados")
            return

        # Prepare sensitive labels (ensure they are on CPU for numpy indexing operations)
        if isinstance(sensitive_labels, torch.Tensor):
            self.sensitive_train = sensitive_labels.cpu().numpy()
        else:
            self.sensitive_train = np.array(sensitive_labels)
            
        unique_groups = np.unique(self.sensitive_train)
        
        # Initialize the meta-model
        self.meta_model = ModeloEnBruto(self.input_dim).to(device)
        meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=self.meta_lr)

        print(f"Starting Meta-Training on {len(unique_groups)} groups...")

        # === META TRAINING LOOP ===
        for epoch in range(self.meta_epochs):
            meta_optimizer.zero_grad()
            meta_loss = 0.0 
            valid_tasks = 0

            # para cada grupo
            for g_id in unique_groups:
                # samplear un batch
                support_x, support_y, query_x, query_y = self._sample_task_batches(g_id)
                
                if support_x is None: continue # Skip groups with too few samples

                # Clone parameters for inner loop
                params = {name: param for name, param in self.meta_model.named_parameters()}

                # entrenar sobre el batch inner_step veces (Inner Loop)
                for _ in range(self.inner_steps):
                    support_logits = self.meta_model(support_x, params)
                    support_loss = self.loss_fn(support_logits, support_y)
                    
                    # Compute gradients
                    grads = torch.autograd.grad(support_loss, params.values(), create_graph=True)
                    
                    # Update parameters (Manual SGD)
                    params = {
                        name: param - self.inner_lr * grad
                        for (name, param), grad in zip(params.items(), grads)
                    }

                # Evaluate on query set
                query_logits = self.meta_model(query_x, params)
                task_loss = self.loss_fn(query_logits, query_y)
                meta_loss += task_loss
                valid_tasks += 1

            if valid_tasks > 0:
                meta_loss = meta_loss / valid_tasks
                meta_loss.backward()
                meta_optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(f"Meta-epoch {epoch+1}/{self.meta_epochs} | meta-loss={meta_loss.item():.4f}")

        # === ADAPTATION PHASE (Fine-tuning for inference) ===
        # After meta-learning, we create specific adapted parameters for each group
        # using their entire training set (or a large support set)
        print("Meta-training finished. Adapting models for each group...")
        
        for g_id in unique_groups:
            # Get all training data for this group to fine-tune
            support_idx = np.where(self.sensitive_train == g_id)[0]
            X_support = self.X_train[support_idx]
            y_support = self.y_train[support_idx]
            
            # Start from meta-learned weights
            # Note: We detach here because we don't need to backprop to meta-model anymore
            adapted_params = {name: param.clone().detach().requires_grad_(True) 
                              for name, param in self.meta_model.named_parameters()}
            
            # Simple optimizer for fine-tuning
            # (Using manual SGD loop to match the functional approach of ModeloEnBruto)
            for _ in range(self.inner_steps):
                logits = self.meta_model(X_support, adapted_params)
                loss = self.loss_fn(logits, y_support)
                
                grads = torch.autograd.grad(loss, adapted_params.values())
                adapted_params = {
                    name: param - self.inner_lr * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }
            
            self.group_params[g_id] = adapted_params

    def predict(self, sensitive_labels=None, **kwargs):
        """
        Returns predictions.
        If sensitive_labels are provided (Test set labels), it uses the group-adapted parameters.
        If not provided, it uses the base meta-model (zero-shot).
        """
        self.meta_model.eval()
        n_samples = self.X_test.shape[0]
        
        # Prepare output tensor
        predictions = torch.zeros(n_samples, device=device)
        
        # If we have group info for the test set, we use the specific adapted models
        if sensitive_labels is not None:
            if isinstance(sensitive_labels, torch.Tensor):
                test_groups = sensitive_labels.cpu().numpy()
            else:
                test_groups = np.array(sensitive_labels)
                
            unique_test_groups = np.unique(test_groups)
            
            for g_id in unique_test_groups:
                mask = (test_groups == g_id)
                mask_tensor = torch.tensor(mask, device=device)
                
                if g_id in self.group_params:
                    # Use adapted parameters
                    X_group = self.X_test[mask_tensor]
                    if X_group.shape[0] > 0:
                        with torch.no_grad():
                            logits = self.meta_model(X_group, params=self.group_params[g_id])
                            predictions[mask_tensor] = torch.sigmoid(logits)
                else:
                    # Fallback if group was not seen in training
                    X_group = self.X_test[mask_tensor]
                    if X_group.shape[0] > 0:
                        with torch.no_grad():
                            logits = self.meta_model(X_group, params=None) # Uses internal weights
                            predictions[mask_tensor] = torch.sigmoid(logits)
        else:
            # No sensitive labels provided for test, use base Meta-Model
            # print("Warning: No sensitive_labels provided for predict. Using base meta-model.")
            with torch.no_grad():
                logits = self.meta_model(self.X_test, params=None)
                predictions = torch.sigmoid(logits)

        # Return mxn matrix (m samples, 1 output) on CPU
        return predictions.unsqueeze(1).cpu().numpy()