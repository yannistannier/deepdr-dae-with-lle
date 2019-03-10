# DeepDr : Apprentissage Profond pour la Reduction de Dimension

L’objectif de ce projet est de contribuer au developpement d’un cadre unifié et flexible pour les approches simultanees qui combinent l’apprentissage profond via une architecture Autoencoder et les techniques de
reduction de dimension telles que LLE (Locally Linear Embedding), ISOMAP, et EIGENMAP..etc.


La méthode proposée, peut etre vue comme une procedure recherchant simultanement une nouvelle representation des donnees contenant le maximum d’informations (en utilisant un DAE ), et un graphe de similarite caracterisant au mieux la proximite entre le points (en utilisant LLE). cette methode consiste dans l’optimisation du probleme suivant : 

![alt text](https://raw.githubusercontent.com/yannistannier/deepdr-dae-with-lle/master/images/1.png)


θ1, θ2 sont respectivement les parametres des blocs encodeur et decodeur de l’AE. S est la matrice des poids caracterisant la proximite entre les poits calculer avec LLE. Cette fonction objectif se decompose en deux
1
termes, le premier correspond a la fonction objectif d’un Autoencodeurs et le second teme correspond a la fonction objectif de la methode LLE.

L'objectif est un algorithme iteratif simple, optimisant une fonction objective appropriee. Cet algorithme s’appuie sur deux etapes de mise a jour selon le schema d'ecrit dans le pseudo-code ci-apre :

![alt text](https://raw.githubusercontent.com/yannistannier/deepdr-dae-with-lle/master/images/2.png)


## Rapport :

[DeepDR_Gesnouin_Tannier.pdf](https://raw.githubusercontent.com/yannistannier/deepdr-dae-with-lle/master/DeepDR_Gesnouin_Tannier.pdf)



## Structure depot :


Main code :  
- [deepReduc.ipynb](deepReduc.ipynb) : Notebook et visuel de l'implementation de DeepDr
- [deepReduc.py](deepReduc.py) : Script DeepDr
- [GMM_nmiari.ipynb](GMM_nmiari.ipynb) : Notebook du script des test des mélanges de modèles gaussiens et des indicateurs classiques NMI/ARI (plus de resultat dans le dossier "GMM_nmi_ari")


Visualisation :
- [ProjetReducDim.r](ProjetReducDim.r) : Comparaison et état de l'art des methodes de réduction de dimension
- [Resultat.ipynb](Resultat.ipynb) : Plot T-SNE des differentes méthodes


Repertoires :
- models => contient les models entrainées de deepDr
- result => contient tout les bottlenecks des differentes configuration effectuées tout au long de ce projet pour chaque dataset : Mnist / fashion mnist / pendigit / upsps


Experimentation : Contient differentes implementations d'ameliorations possible.
- [deepR_denoising.py](experimentation/deepR_denoising.py) : Denoising DeepDr
- [VAE-DeepDr.ipynb](experimentation/VAE.ipynb) : Variational DeepDr & plot T-SNE (comparatif)
- [deepR-vae-bvae.py](experimentation/deepR-vae-bvae.py) : Disentangled DeepDr (beta-vae)
- [deepR_sparcity.py](experimentation/deepR_sparcity.py) : DeepDr with sparcity constraint
- [ConvAE.ipynb](experimentation/ConvAE.ipynb) : Convolutional DeepDr
