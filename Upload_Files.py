import gdown
import os
import shutil

PATHS = {os.path.join("Tache3_Analyse_opinion", "Transformer", "Model", "Model"):["training_args.bin",
                                                                                  "pytorch_model.bin",
                                                                                  "config.json"],
         os.path.join("Tache3_Analyse_opinion", "Bi-LSTM", "Model"):["bidirectional_lstm_NN.h5"],
         os.path.join("Tache3_Analyse_opinion", "CNN", "Model"):["CNN.h5"],
         os.path.join("Tache4_barre_de_recherche","model","LSTM","Model"):["bidirectional_lstm_NN_ranking.h5"],
         os.path.join("Tache3_Analyse_opinion", "train"):["training_noemoticon.csv"]}

FOLDER_NAME = "Model"
URL = "https://drive.google.com/drive/folders/1szZfxCGsK3XykrVHzD5D7B6R4a3bNf07?usp=share_link"
def download_file_from_google_drive(url):
    if url.split('/')[-1] == '?usp=sharing':
        url = url.replace('?usp=sharing', '')
    gdown.download_folder(url)




if __name__ == "__main__":
    for path in PATHS:
        if not os.path.exists(path):
          os.makedirs(path, exist_ok=True)

    download_file_from_google_drive(URL)
    for to, names in PATHS.items():
       for name in names:
          shutil.move(os.path.join(FOLDER_NAME, name), os.path.join(to, name))
    shutil.rmtree(FOLDER_NAME)


