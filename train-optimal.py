from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from models import MLP, RNN, Transformer
from training import train

MLP_to_train = MLP(n=16)
RNN_to_train = RNN(hidden_size=16, num_layers=2)
Transformer_to_train = Transformer(n=27, encode_pos=True, model_dimensionality=32, num_of_transformer_blocks=2)

MLP_to_train.__name__ = "MLP"
RNN_to_train.__name__ = "RNN"
Transformer_to_train.__name__ = "Transformer"

MLP_Model = train(model=MLP_to_train, batch_size=1, epochs=100, learning_rate=0.001, window_size=16, optimizer=Adam)
RNN_Model = train(model=RNN_to_train, batch_size=1, epochs=100, learning_rate=0.001, window_size=8, optimizer=Adam)
Transformer_Model = train(model=Transformer_to_train, batch_size=1, epochs=100, learning_rate=0.001, optimizer=AdamW)
