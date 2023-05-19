# ProjetX_V2

L'étude décrite dans le cadre de ce projet vise à extraire les principaux sujets abordés à partir d'un corpus de textes, et à caractériser ces sujets selon différentes dimensions telles que l'opinion, la source, le thème et la localisation. L'objectif est de fournir des outils permettant d'accéder aux contenus correspondants afin de répondre aux besoins d'information exprimés par les utilisateurs, en langage naturel.

Cette étude peut être appliquée dans divers contextes professionnels nécessitant l'exploration de grandes quantités de données textuelles, qu'il s'agisse de textes provenant des réseaux sociaux, du patrimoine documentaire des entreprises, d'enquêtes publiques, etc. L'objectif est d'accomplir des tâches d'analyse sémantique et de fournir des informations compréhensibles aux utilisateurs finaux.

Le contexte scientifique de cette étude est lié à l'analyse du langage naturel, à la recherche d'information et à l'extraction d'information à partir de textes. L'utilisation d'outils d'apprentissage statistique est également fortement recommandée pour mener à bien cette étude.

On utilisera le corpus des tweets de Donald Trump (fichier .csv en anglais) contient près de 30 000 tweets provenant du compte de Donald Trump, datant de 2009 à janvier 2021. Les informations disponibles pour chaque tweet comprennent la source, le texte du tweet, la date du tweet, ainsi que le nombre de retweets et de favoris. Vous pouvez télécharger les données à l'adresse suivante : Trump Twitter Archive.


ce projet se constitue de quatre tâches:
 - **Tâche 1 Constitution, indexation et exploration du corpus** : Créer un fichier inverse du corpus et générer des index pour les données structurées épurées disponibles, incluant la localisation, la catégorie thématique, les favoris, etc
 - **Tâche 2 Analyse thématique du corpus**: Découvrir les thèmes du corpus en s’appuyant sur un modèle non supervisé
 - **Tâche 3 : Analyse d'opinions** : Extraire les opinions des tweets, en employant trois modèles neuronaux (CNN, LSTM, BERT) 
 - **Tâche 4 : Recherche d’informations dans le corpus** : développer une barre de recherche sur le corpus, avec deux modèles (LSTM, BERT)

_note 1_: 
---
Une fois que vous avez cloné le référentiel Git, vous devez exécuter la commande suivante pour télécharger [les modèles de réseaux neuronaux](https://drive.google.com/drive/folders/1szZfxCGsK3XykrVHzD5D7B6R4a3bNf07?usp=share_link):

```python
python .\Upload_Files.py
```
_note 2_ : 
---
Pour la tâche 3 d'analyse d'opinions, afin d'éviter les problèmes de conflit et de compatibilité, il est recommandé d'exécuter les modèles ou de les entraîner sur des environnements virtuels distincts. Il est important de noter que les modèles utilisant BERT requièrent l'utilisation de **PyTorch**, tandis que les modèles CNN et LSTM nécessitent **TensorFlow**. En utilisant des environnements virtuels séparés pour chaque framework, vous pouvez garantir une exécution fluide et sans conflit pour les différents modèles.



