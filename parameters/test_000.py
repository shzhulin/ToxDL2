import torch
import time
from pathlib import Path

HERE = Path("../")
DATA = HERE / "data"
MODEL = HERE / "checkpoints"
OUTPUT = HERE / "predictions"
model_best_save_path = MODEL / ('best_trained_model_{0}'.format(time.strftime('%m%d-%H%M%S')) + '.pth')
test_predicted_save_path = OUTPUT / ('test_predict_result_{0}'.format(time.strftime('%m%d-%H%M%S')) + '.txt')
independent_predicted_save_path = OUTPUT / ('independent_predict_result_{0}'.format(time.strftime('%m%d-%H%M%S')) + '.txt')

# parameters
batch_size = 64
learning_rate = 1e-3
step_size = 10
gamma = 0.1
epochs = 20
device = torch.device('cuda:0')