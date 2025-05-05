
import numpy as np

def get_word_file():

    WordFilePath = '../retinal_dataset/word_embbeding/retinal_word_embedding_300_expert_knowledge_DR5.npy'      
    WordFile = np.load(WordFilePath)

    return WordFile


