import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from enum import Enum

# This is a placeholder, assuming you have a constants file
# If not, you can define the list directly, e.g., GLM_HMM_SIMULUS_VALUES = [-1, -0.5, ..., 1]
try:
    from utils.constants import GLM_HMM_SIMULUS_VALUES
except ImportError:
    # Define a default if the constant is not found, for standalone execution
    GLM_HMM_SIMULUS_VALUES = np.array([-1., -0.5, -0.25, -0.125, -0.0625, 0., 0.0625, 0.125, 0.25, 0.5, 1.])


class CognitiveModel(Enum):
    PRL4 = 1
    PRL_foraging_dynamic = 2
    HRL2 = 3
    GLM_HMM = 4

def get_latent_labels(data, latent_key):
    """Extracts and reshapes latent variable data from a DataFrame."""
    num_agents = len(data['agentid'].unique())
    num_trial = len(data['trials'].unique())
    return data[latent_key].to_numpy().astype(np.float32).reshape((num_agents, num_trial))

def get_labels_by_model(data, model: CognitiveModel):
    """
    Extracts and formats the target labels based on the specified cognitive model.
    Converted to use PyTorch for one-hot encoding where necessary.
    """
    n_agent = len(data['agentid'].unique())
    n_trial = len(data['trials'].unique())
    
    if model == CognitiveModel.PRL4:
        # Continuous label
        qv = np.array(data['rewards'] - data['rpe_history'])
        return qv.astype(np.float32).reshape((n_agent, n_trial))
        
    elif model == CognitiveModel.PRL_foraging_dynamic:
        # Discrete label
        state_labels = get_latent_labels(data, ['latent_att'])
        n_st = len(data.latent_att.unique())
        # PyTorch one-hot encoding
        state_labels_tensor = torch.tensor(state_labels, dtype=torch.long)
        normalized_st_labels = F.one_hot(state_labels_tensor, num_classes=n_st).float().numpy()
        
        # Continuous label
        winning_q_labels = get_latent_labels(data, ['chosen_qv'])
        return winning_q_labels, normalized_st_labels
        
    elif model == CognitiveModel.HRL2:
        # Discrete label
        chosen_cue_labels = get_latent_labels(data, ['chosencue'])
        n_cue = len(data.chosencue.unique())
        # PyTorch one-hot encoding
        chosen_cue_tensor = torch.tensor(chosen_cue_labels, dtype=torch.long)
        normalized_cue_labels = F.one_hot(chosen_cue_tensor, num_classes=n_cue).float().numpy()
        
        # Continuous label
        qv = np.array(data['rewards'] - data['rpe_history'])
        winning_q_labels = qv.astype(np.float32).reshape((n_agent, n_trial))
        return winning_q_labels, normalized_cue_labels
        
    elif model == CognitiveModel.GLM_HMM:
        # Discrete label
        state_labels = get_latent_labels(data, 'which_state')
        n_st = len(data.which_state.unique())
        # PyTorch one-hot encoding
        state_labels_tensor = torch.tensor(state_labels, dtype=torch.long)
        return F.one_hot(state_labels_tensor, num_classes=n_st).float().numpy()

def get_feature_list_by_model(model: CognitiveModel):
    """Returns the list of feature columns for a given model."""
    if model == CognitiveModel.PRL4:
        return ["actions", "rewards"]
    elif model == CognitiveModel.PRL_foraging_dynamic:
        return ["actions", "rewards"]
    elif model == CognitiveModel.HRL2:
        return ["chosenside", "rewards", "allstims0", "allstims1", "allstims2"]
    elif model == CognitiveModel.GLM_HMM:
        return ["stim", "actions", "wsls"]

def get_onehot_features(data, input_list):
    """
    Converts specified feature columns into a single one-hot encoded tensor.
    Converted to use PyTorch for one-hot encoding and concatenation.
    """
    n_agent = len(data["agentid"].unique())
    n_trial = len(data["trials"].unique())
    features = []
    
    for key in input_list:
        input_data = data[key].to_numpy()
        unique_input = GLM_HMM_SIMULUS_VALUES if key == "stim" else np.unique(input_data)
        
        # Create a mapping from unique value to integer index
        cat_map = {item: i for i, item in enumerate(unique_input)}
        input_cat = [cat_map[s] for s in input_data]
        
        # Convert to tensor and one-hot encode
        input_tensor = torch.tensor(input_cat, dtype=torch.long).reshape((n_agent, n_trial))
        one_hot_tensor = F.one_hot(input_tensor, num_classes=len(unique_input)).float()
        features.append(one_hot_tensor)
        
    # Concatenate all feature tensors along the last dimension
    return torch.cat(features, dim=2)
