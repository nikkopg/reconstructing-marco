from marco import MarcoWrapper
from utils import *
from test import count_matched_weights

# Loading Marco old savedmodel/checkpoint
checkpoint_path = 'old_marco_model/variables/variables'
checkpoint_vars = load_checkpoint_variables(checkpoint_path)

# Instantiate reconstructed Marco
reconstructed_marco = MarcoWrapper(depth_multiplier=1.0)
reconstructed_marco.load_weights(checkpoint_variables=checkpoint_vars)
print(reconstructed_marco.model.summary())

# Check if the assigned weights matched with the old checkpoints
matched_counts = count_matched_weights(checkpoint_vars, reconstructed_marco)
print(f"Total matched: {matched_counts}/{len(checkpoint_vars)}")