# from rasa_nlu.converters import load_data
# from rasa_nlu.config import RasaNLUConfig
# from rasa_nlu.model import Trainer
#
# def train_nlu(data, config, model_dir):
#     training_data = load_data(data)
#     trainer = Trainer(RasaNLUConfig(config))
#     trainer.train(training_data)
#     model_directory = trainer.persist(model_dir,fixed_model_name = 'weathernlu')
#
# if __name__ == '__main__':
#     train_nlu('./data/data.json','config_spacy.json','./models/nlu')
from rasa_nlu.training_data import load_data
# from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Metadata, Interpreter

def train_nlu():
    training_data = load_data('./data/data.json')
    trainer = Trainer(config.load('config_spacy.json'))
    trainer.train(training_data)
    model_directory = trainer.persist('./models/nlu' ,  fixed_model_name = 'myfirstbot')

def run_nlu():
    interpreter = Interpreter.load('./models/nlu/default/myfirstbot')
    print(interpreter.parse("What's the weather in Berlin at the moment?"))

if __name__ == '__main__':
    run_nlu()
    # run_nlu()
