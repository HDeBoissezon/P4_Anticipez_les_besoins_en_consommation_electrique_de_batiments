import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
import timeit

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
#                 plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()


def filtre_IQR(df, liste_colonnes):
    """[summary]
    fonction pour remplacer les outliers par des nan

    Args:
        df (pd.DataFrame): [description]
        col_num (list): [description]
    """
    for col in liste_colonnes:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3-Q1
        print(col, ': suppression des valeurs inférieures à ', (Q1 - 1.5 * IQR), 'et supérieures à ', (Q3 + 1.5 * IQR))
        df.loc[:,col] = df[col].where(cond=(df[col] > (Q1 - 1.5 * IQR)) & (df[col] < (Q3 + 1.5 * IQR)), other=np.nan)  
    return df  


def pourcent_NA_df(df):
    """ 
    affichage du taux de remplissage global d'un dataframe
    """
    print("Il y a {:0.2f}% valeurs renseignées (soit {:0.2f} % de valeurs manquantes) dans l'intégralité du df".format(100-100*df.isna().mean().mean(), 100*df.isna().mean().mean()))

    
def taux_remplissage_colonne(df):
    """
    Calcul du taux de remplissage en % pour chaque colonne 
    """
    NA = 100 - 100*df.isna().mean(axis=0)
    
    # transformation en df
    NA_df = pd.DataFrame(NA, columns=['taux remplissage'])
    NA_df = NA_df.sort_values(by='taux remplissage', ascending=False).reset_index()
    
    return NA_df    
    
def visu_remplissage_colonnes(df, seuil):
    """"
    visualisation du taux de remplissage et affichage d'un seuil
    arg :
        df = dataframe original
        seuil = valeur (exprimée en %) de seuil à afficher
    """
    
    # calcul du taux de remplissage des colonnes
    NA_df = taux_remplissage_colonne(df)
    
    hauteur = int(NA_df.shape[0]/4)
    fig = plt.figure(figsize=(8, hauteur))
    sup_threshold = seuil
    font_title = {'family': 'serif',
              'color':  '#114b98',
              'weight': 'bold',
              'size': 18,
             }

    sns.barplot(x='taux remplissage', y='index', data=NA_df, palette="flare")
    plt.axvline(x=sup_threshold, linewidth=2, color = 'r')
    plt.text(sup_threshold+2, 45, 'Seuil de {:.0f}%'.format(sup_threshold), fontsize = 16, color = 'r')

    plt.title("Taux de remplissage des variables dans le jeu de données (%)", fontdict=font_title)
    plt.xlabel("Taux de remplissage (%)")
    plt.show()

def Affichage_duree(start_time):
    '''
    Fonction pour afficher les durées de calcul de façon plus claire
    '''
    elapsed = timeit.default_timer() - start_time
    if elapsed < 60:
        print(f'Temps d\'exécution : {elapsed:.2f}s')
    elif elapsed < 3600:
        elapsed_m = elapsed/60
        print(f'Temps d\'exécution : {elapsed_m:.2f}min')
    else:
        elapsed_h = elapsed/3600
        print(f'Temps d\'exécution : {elapsed_h:.2f}h')
    return elapsed
    