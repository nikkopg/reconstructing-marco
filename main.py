from marco import Marco
from utils import *
from test import count_matched_weights

checkpoint_path = 'models/google_model/1/variables/variables'
checkpoint_vars = load_checkpoint_variables(checkpoint_path)

model = Marco()

matched_counts = count_matched_weights(checkpoint_vars, model)
print(f"Total matched: {matched_counts}/{len(checkpoint_vars)}")