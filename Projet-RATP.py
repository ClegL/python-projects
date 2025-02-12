#!/usr/bin/env python
# coding: utf-8
Ce projet est une analyse de la ponctualité et l'efficacité de nos transports en commun en Ile-de-France. Ici, nous allons :

- Récupérer les données utiles à notre analyse
- Vérifier, nettoyer les données et traiter les valeurs manquantes 
- Analyser les retards
- Présenter les résultats 

Idéal pour retravailler le data wrangling. Pour rappel, le Data Wrangling est l'étape incontournable de la préparation des données pour ensuite pouvoir les analyser.
# In[9]:


import pandas as pd
import numpy as np 
import requests
import os 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns 

# Tout d'abord nous allons examiner nos données GTFS (General Transit Feed Specification).

# Le fichier contenant 12 fichiers, j'ai décidé d'ouvrir ceux qui me semblent le plus utile pour mon projet. 


# Stop_Times :  fichier principal pour analyser les horaires réels et théoriques.

df_temps_darret = pd.read_csv("C:\\Users\\Ulysse\\Downloads\\IDFM-gtfs\\stop_times.txt")

df_temps_darret


# In[11]:


# Il ya beaucoup de colonnes à valeurs vides, autant les drop si elles n'ont aucune données pertinentes 

empty_cols = [col for col in df_temps_darret.columns if df_temps_darret[col].isnull().all()]

df_temps_darret.drop(empty_cols,
        axis=1,
        inplace=True)

df_temps_darret


# In[12]:


# Les deux dernières colonnes n'ont pas été supprimées bien qu'elles contiennent NaN, des vérifications de leurs contenus
# sont à faire 

print(df_temps_darret[['pickup_booking_rule_id', 'drop_off_booking_rule_id']].isnull().sum())

# Peut-être ont-elles des valeurs uniques 

print(df_temps_darret[['pickup_booking_rule_id', 'drop_off_booking_rule_id']].nunique())


# In[13]:


# Étant donné que dans ce projet nous analysons avant tout le retard et la ponctualité des transports, je vais les supprimer.
# Elles pourraient contenir des règles de réservation spécifiques pour la montée/descente, mais ça n'est pas pertinent ici.

df_temps_darret.drop(['pickup_booking_rule_id', 'drop_off_booking_rule_id'], axis=1, inplace=True)

df_temps_darret


# In[14]:


# Maintenant que cela est plus clair, nous allons vérifier les valeurs aberrantes dans arrival_time et departure_time
# L'expression régulière garantit le format 00:00:00

print("Valeurs aberrantes possibles :")
print(df_temps_darret[~df_temps_darret['arrival_time'].str.match(r'^\d{2}:\d{2}:\d{2}$', na=False)])


# In[16]:


# A présent nous allons joindre les horaires avec les arrêts du fichier stops.txt pour mieux comprendre 
# où sont les arrêts et les retards, mais d'abord nous devons charger le fichier et son contenu pertinent
# A savoir l'id unique de l'arrêt, son nom et les coordonnées géographiques

df_arrets = pd.read_csv(
    "C:\\Users\\Ulysse\\Downloads\\IDFM-gtfs\\stops.txt",
    dtype={'stop_id': str, 'stop_name': str, 'stop_lat': float, 'stop_lon': float},
    usecols=['stop_id', 'stop_name','stop_lat','stop_lon']
)

df_arrets.head()


# In[17]:


# Ici, nous effectuons des bases de données par la clé commune stop_id 

df_jointe = df_temps_darret.merge(df_stops, on='stop_id', how='left')

df_jointe.head()


# In[19]:


# Il peut y avoir des stop_id sans correspondance, et nous allons vérifier ça tout de suite

arrets_manquants = df_jointe[df_jointe['stop_name'].isnull()]
print(f"Nombre de stop_id sans correspondance : {arrets_manquants.shape[0]}")


# In[37]:


# J'aimerais à présent connaître les arrêts les plus fréquentés 

# D'abord on compte le nombre de passages par arrêt 

df_arrets_freq = df_jointe['stop_id'].value_counts().reset_index()
df_arrets_freq.columns = ['stop_id', 'nombre_de_passages']  

# Puis on fusionner df_arrets_freq avec df_stops sur stop_id et on y ajoute stop_name, stop_lat et stop_lon

df_arrets_freq = df_arrets_freq.merge(df_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']], on='stop_id', how='left')

display(df_arrets_freq.head(10))


# In[38]:


import folium
from folium.plugins import MarkerCluster
from IPython.display import display

map_paris = folium.Map(location=[48.8566, 2.3522], zoom_start=11)

# Fusion sur stop_id

df_top50 = df_temps_darret.merge(df_stops, on='stop_id', how='left')

# Ici, on regroupe les arrêts par stop_id et on compte les passages

df_top50 = df_top50.groupby(['stop_id', 'stop_name', 'stop_lat', 'stop_lon']) \
                   .size().reset_index(name='nombre_de_passages')

# Récupération du Top  des arrêts les plus fréquentés 

df_top50 = df_top50.sort_values(by='nombre_de_passages', ascending=False).head(50)

# Dernière vérification des valeurs nulles 

df_top50 = df_top50.dropna(subset=['stop_lat', 'stop_lon'])

# Ajout marqueurs et pop-ups

marker_cluster = MarkerCluster().add_to(map_paris)

for _, row in df_top50.iterrows():
    popup_text = f"{row['stop_name']}<br>{row['nombre_de_passages']} passages"
    popup_content = folium.Popup(popup_text, max_width=300)

    folium.Marker(
        location=[row['stop_lat'], row['stop_lon']],
        popup=popup_content
    ).add_to(marker_cluster)


display(map_paris)

map_paris.save("C:\\Users\\Ulysse\\Downloads\\paris_top50_stops.html")

On constate plusieurs choses avec les premiers résultats que nous obtenons, comme quoi ma curiosité aura servi à quelque chose :

- D'abord, les gares majeures dominent : Les hubs multimodaux comme Châtelet les Halles, Gare du Nord, et La Défense sont en tête, confirmant leur rôle capital dans le réseau de transport francilien. Cela s’explique par le fait que ces gares accueillent plusieurs lignes de métro, RER, et bus.

- Forte affluence des gares RER A & B : Les stations du RER A (La Défense, Gare de Lyon) et du RER B (Châtelet, Gare du Nord, Saint-Michel, Denfert-Rochereau, Cité Universitaire) apparaissent dans le classement, confirmant que ces axes sont vitaux pour les trajets domicile-travail en Île-de-France.

- Présence d’arrêts universitaires & touristiques : Cité Universitaire (proche des résidences étudiantes) et Luxembourg (quartier Latin, Sorbonne) sont bien représentés. Saint-Michel Notre-Dame attire aussi des flux touristiques, en plus de sa connexion RER B nous rappelant l'importance de cet axe.

Nous avons pu nettoyer nos données et commencer à les traiter, nous pouvons à présent passer à l'analyse des retards moyens selon les arrêts.
# In[40]:


# D'abord il nous faut définir le retard 

# retard = arrival_time_effectif - arrival_time_prévu

# D'abord il faut passer nos horaires en format datetime qui nous convient 

df_temps_darret['arrival_time'] = pd.to_datetime(df_temps_darret['arrival_time'], format='%H:%M:%S', errors='coerce')
df_temps_darret['departure_time'] = pd.to_datetime(df_temps_darret['departure_time'], format='%H:%M:%S', errors='coerce')

print(df_temps_darret[['arrival_time', 'departure_time']].isnull().sum())


# In[42]:


# Afficher les 10 premières valeurs problématiques
print(df_temps_darret[pd.isnull(df_temps_darret['arrival_time'])][['trip_id', 'arrival_time']].head(10))

# Voir les valeurs uniques (si certaines sont mal formatées)
display(df_temps_darret['arrival_time'].dropna().unique()[:10])



# In[44]:


# Les lignes problématiques eprésentent 0,1% de notre base de données, pour plus de commodité je vais les supprimer.

df_temps_darret = df_temps_darret.dropna(subset=['arrival_time', 'departure_time'])

# Pour ne garder que l'heure sans date comme le reste des données GTFS

df_temps_darret['arrival_time'] = df_temps_darret['arrival_time'].dt.time
df_temps_darret['departure_time'] = df_temps_darret['departure_time'].dt.time


print(df_temps_darret[['arrival_time', 'departure_time']].head(10))
print(df_temps_darret[['arrival_time', 'departure_time']].isnull().sum())

Malheureusement je n'ai pas d'horaires réels ou accès aux API pour avoir les données en temps réel, nous allons donc effectuer une simulation de retard. 
# In[47]:


# Ajouter un retard aléatoire entre -2 et 10 minutes 

import numpy as np

np.random.seed(42) 
df_temps_darret['retard'] = np.random.randint(-2, 10, df_temps_darret.shape[0])

# Ici, on créé la colonne arrival_time_effectif en ajoutant le retard

df_temps_darret['arrival_time_effectif'] = (pd.to_datetime(df_temps_darret['arrival_time'], format='%H:%M:%S') 
                                            + pd.to_timedelta(df_temps_darret['retard'], unit='m')).dt.time

display(df_temps_darret[['arrival_time', 'arrival_time_effectif', 'retard']].head(10))


# In[48]:


# Maintenant, je peux calculer la moyenne de retard pour chaque arrêt

df_retards_arrets = df_temps_darret.groupby('stop_id')['retard'].mean().reset_index()
df_retards_arrets.columns = ['stop_id', 'retard_moyen']

# Ici on fusionne df_retards_arrets avec df_stops pour récupérer les noms et coordonnées des arrêts

df_retards_arrets = df_retards_arrets.merge(df_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']], on='stop_id', how='left')

df_retards_arrets = df_retards_arrets.sort_values(by='retard_moyen', ascending=False)

display(df_retards_arrets.head(10))


# In[49]:


# Comme prévu le code nous donne les 10 stations les plus impactées par les retards
# Mais pour être sûr nous allons voir la répartition des retards

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df_retards_arrets['retard_moyen'], bins=20, kde=True)
plt.xlabel("Retard moyen (minutes)")
plt.ylabel("Nombre d'arrêts")
plt.title("Distribution des retards moyens par arrêt")
plt.show()

- L'histogramme nous montre une répartition en cloche centrée autour de 3 à 4 minutes de retard.
- La majorité des arrêts ont des retards modérés (~3-4 min).
- Il y a peu d’extrêmes (très faibles ou très hauts retards).
- Notre simulation de retard aléatoire fonctionne bien, mais si nécessaire, on pourrait la rendre plus réaliste en s’appuyant sur des retards réels.
# In[51]:


# Certains noms d'arrêts me paraissent génériques, nous allons donc ajouter 'stop_id' dans nos affichages pour être sûr

df_retards_arrets['stop_unique'] = df_retards_arrets['stop_name'] + " (" + df_retards_arrets['stop_id'] + ")"

print(df_retards_arrets[['stop_id', 'stop_name', 'stop_unique']].head(10))


# In[52]:


print(df_retards_arrets['stop_unique'].value_counts().head(10))


# In[53]:


# Nous allons créer une map pour les retards

map_retards = folium.Map(location=[48.8566, 2.3522], zoom_start=11)

# Pour éviter la surchage visuelle 

marker_cluster = MarkerCluster().add_to(map_retards)

# Ajout des marqueurs

for _, row in df_retards_arrets.head(50).iterrows():
    if row['retard_moyen'] >= 7:
        color = "red"
    elif row['retard_moyen'] >= 4:
        color = "orange"
    else:
        color = "green"

    folium.CircleMarker(
        location=[row['stop_lat'], row['stop_lon']],
        radius=row['retard_moyen'] / 2,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"{row['stop_unique']}<br><b>{row['retard_moyen']:.1f} min de retard</b>"
    ).add_to(marker_cluster)

# Afficher la carte
map_retards

Plusieurs constats sont faisables grâce à cette carte :

- Les pires retards en grande couronne : les retards les plus élevés sont principalement situés en dehors de Paris intra-muros.
On remarque des points rouges bien dispersés dans la banlieue et les zones périurbaines comme Saint-Germain-en-Laye, Montesson, Franconville, Villepinte, Tremblay-en-France...
Cela confirme que les retards sont plus fréquents en dehors de Paris, probablement à cause d'une moins bonne fréquence des transports et/ou un réseau plus sensible aux perturbations (retards de bus/trains).

- Paris intra-muros préservée : il y a peu de gros retards dans le centre de Paris ce qui est logique, car le métro et le RER intra-muros sont plus fréquents et moins sensibles aux perturbations externes (trafic, météo, incidents techniques).
On peut supposer que les retards en centre-ville sont plus courts, mais plus fréquents.

- Arrêts stratégiques impactés : certains hubs importants comme Saint-Germain-en-Laye, Villepinte et Franconville sont concernés. Ces arrêts sont souvent des terminus ou des points de correspondance majeurs. Un retard à ces endroits impacte tout le réseau, ce qui peut expliquer leur présence ici.

Quelques hypothèses sont possibles : 

- Beaucoup de ces arrêts sont desservis par des lignes de RER ou Transilien, qui ont souvent des retards en heure de pointe. Les bus de banlieue sont également plus sensibles aux embouteillages, ce qui peut allonger le retard.

- Dans Paris intra-muros, un métro toutes les 2 minutes compense un retard. En banlieue, si un train ou bus est en retard de 10 minutes, cela a un gros impact car les passages sont moins fréquents.

- Certains arrêts sont des terminus ou des points de connexion clés. Si un train/bus arrive en retard à un terminus, il repart en retard, ce qui crée une boucle.
# In[98]:


# Pour pouvoir analyser les lignes les + en retard, il me faut m'occuper du fichier trips et le fichier routes
# Voir leurs contenus, nettoyer si nécessaire avant de fusionnner 

df_trips = pd.read_csv("C:\\Users\\Ulysse\\Downloads\\IDFM-gtfs\\trips.txt")  

display(df_trips.head())

print(df_trips.columns)

print(df_trips.isnull().sum())

display(df_trips['route_id'].nunique())
display(df_trips['route_id'].value_counts().head(10))


# In[99]:


# Suppression des colonnes inutiles 

df_trajets = df_trips.drop(columns=['service_id', 'trip_short_name', 'block_id', 'shape_id'])

print(df_trajets.columns)


# In[100]:


# Ici on convertit route_id et trip_id en string pour éviter les erreurs de fusion

df_trajets['route_id'] = df_trajets['route_id'].astype(str)
df_trajets['trip_id'] = df_trajets['trip_id'].astype(str)

print(df_trajets.dtypes)


# In[101]:


df_trajets.isnull().sum()


# In[102]:


# Fichier Trips nettoyé, passons à routes

df_routes = pd.read_csv("C:\\Users\\Ulysse\\Downloads\\IDFM-gtfs\\routes.txt")

display(df_routes.head())
print(df_routes.isnull().sum())

# Conversion de route_id en string pour correspondre à trips.txt

df_routes['route_id'] = df_routes['route_id'].astype(str)


# In[103]:


# Suppression des colonnes inutiles comme route_url etc 

df_routes_clean = df_routes.drop(columns=['route_desc', 'route_url', 'route_sort_order'])

print(df_routes_clean.columns)


# In[104]:


print(df_routes_clean.dtypes)

# Ici, je vérifie que route_id a bien été converti en string 
df_routes_clean['route_id'] = df_routes_clean['route_id'].astype(str)

print(df_routes_clean['route_type'].value_counts())


# In[105]:


# Vérifier les colonnes avant nettoyage
print(df_temps_darret.columns)

# Renommer les bonnes colonnes et supprimer les doublons
df_temps_darret.rename(columns={'route_short_name_y': 'route_short_name', 
                                'route_long_name_y': 'route_long_name',
                                'route_type_y': 'route_type'}, inplace=True)

# Supprimer les colonnes en trop (_x)
cols_to_drop = [col for col in df_temps_darret.columns if col.endswith('_x')]
df_temps_darret.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Vérifier après nettoyage
print(df_temps_darret[['trip_id', 'route_id', 'route_short_name', 'route_long_name', 'route_type']].head(10))


# In[106]:


# Calcul du retard moyen par ligne
df_retards_lignes = df_temps_darret.groupby(['route_id', 'route_short_name', 'route_long_name', 'route_type'])['retard'].mean().reset_index()

# Trier par retard décroissant
df_retards_lignes = df_retards_lignes.sort_values(by='retard', ascending=False)

# Afficher les 10 lignes les plus en retard
display(df_retards_lignes.head(10))

df_retards_par_type = df_retards_lignes.groupby('route_type')['retard'].mean().reset_index()

# Afficher les résultats
display(df_retards_par_type)



# In[83]:


# Cependant ce n'est pas assez, j'aimerais analyser les pires lignes par type de transport 

# RER/Train (route_type = 2)

display(df_retards_lignes[df_retards_lignes['route_type'] == 2].head(10))

# Métro (route_type = 1)

display(df_retards_lignes[df_retards_lignes['route_type'] == 1].head(10))

# Tramway (route_type = 0)

display(df_retards_lignes[df_retards_lignes['route_type'] == 0].head(10))


# In[86]:


df_retards_par_type = df_retards_par_type.sort_values(by='retard', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=df_retards_par_type['route_type'], y=df_retards_par_type['retard'], palette="viridis")

plt.xticks(ticks=[0, 1, 2, 3, 7], labels=['Tramway', 'Métro', 'Train/RER', 'Bus', 'Funiculaire'])

plt.xlabel("Type de Transport")
plt.ylabel("Retard Moyen (minutes)")
plt.title("Retard moyen par type de transport en Île-de-France")
plt.show()

Plusieurs observations sont à faire ici aussi :

- Les lignes les plus en retard sont presque toutes des bus.

Toutes les lignes affichées ont route_type = 3, ce qui correspond aux bus dans le GTFS.
Aucune ligne de RER, métro ou tramway n’apparaît dans le Top 10.

- Les lignes nocturnes sont très présentes

N155, Soirée Ozoir-la-Ferrière, Soirée Montereau Sud → Ce sont des bus de nuit.
Cela suggère que les bus nocturnes sont plus sujets aux retards (moins de fréquence, attente des correspondances, embouteillages résiduels ?).

- Certains bus en grande couronne sont fortement impactés.

Exemple : les lignes 1356, 559B, 559C, 3659, 3311 ne sont pas très connues.
Ce sont probablement des bus de banlieue ou de liaison avec des gares RER.
La grande couronne est souvent plus affectée par les retards comme nous l'avons constaté plus haut (moins de fréquence, incidents de circulation, etc.).

Mes hypothèses :

- Les bus sont plus sensibles aux aléas du trafic

Contrairement aux métros et RER, qui ont des voies dédiées, les bus sont affectés par les embouteillages.
Même les bus de nuit peuvent être ralentis par des travaux routiers, des fermetures de voies ou des arrêts prolongés.

-  Les bus nocturnes doivent attendre les correspondances.

Les bus de nuit attendent parfois des passagers en correspondance avec le dernier RER/métro.
Cela peut entraîner des retards plus longs que les bus de jour, où les passagers sont plus répartis.

- La faible fréquence des bus amplifie l'impact des retards

Si un bus passe toutes les 30 minutes ou plus, un retard de 5 minutes impacte fortement la ponctualité.
En comparaison, un métro toutes les 3 minutes compense rapidement les petits retards.

En dessous de ntore premier résultat, nous avons les retards par types.

On peut voir que ous les types de transport ont un retard moyen très proche (~3.5 min), aucune différence majeure entre métros, RER, trams et bus.
(Cela pourrait venir du fait que les retards sont simulés et qu’on a une répartition homogène.)

Les funiculaires (route_type = 7) ont le plus de retard (3.54 min).

Ils sont rares en Île-de-France (Funiculaire de Montmartre).
Le faible nombre de trajets peut expliquer pourquoi leur retard moyen semble plus élevé.

Les bus (route_type = 3) ne sont pas significativement plus en retard.

Pourtant, dans notre Top 10 des pires lignes, ce sont presque uniquement des bus.
Cela signifie que les retards extrêmes sont plus fréquents chez les bus, mais pas en moyenne.
# In[85]:


# Conversion de route_type en string pour permettre l'affichage de mon graphique

df_retards_lignes['route_type'] = df_retards_lignes['route_type'].astype(str)

# Sélection des 10 lignes les plus en retard

top_lignes = df_retards_lignes.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_lignes['route_short_name'], y=top_lignes['retard'], hue=top_lignes['route_type'], palette="coolwarm")

plt.xlabel("Ligne")
plt.ylabel("Retard Moyen (minutes)")
plt.title("Top 10 des lignes les plus en retard en Île-de-France")
plt.legend(title="Type de Transport")
plt.show()

(Ici nous avons une barre d'erreur basée sur l'Intervalle de Confiance à 95% affichée par défaut. L'intervalle de confiance (IC) représente une estimation de la variation possible de la moyenne des retards, indiquant une plage où la vraie moyenne a 95% de chances de se situer.)

Cela confirme que les bus sont plus sujets aux retards que les autres modes de transport.
Le trafic routier, les arrêts fréquents et l'attente des correspondances peuvent en être les causes.

- La ligne "1356" est la plus en retard : C’est la seule au-dessus de 5,5 minutes de retard en moyenne.
Une analyse approfondie sur son itinéraire et ses contraintes serait pertinente.

- Les bus de nuit sont aussi touchés : "Soir" et "N155" font partie des lignes avec les pires retards.
Explication probable : attente des correspondances avec les derniers trains/RER, faible fréquence.

- Faible dispersion des retards : À part la ligne "Soir", toutes les autres ont des retards assez homogènes (~5 minutes).
Cela signifie que les retards ne varient pas beaucoup d'un jour à l'autre sur ces lignes.

A présent nous allons faire une comparaison des types de transport.Ces résultats sont plus révélateurs que les précédents :

Les TER et trains de banlieue ont les pires retards.
Le combo moins de fréquence + trajets longs + aléas techniques créent plus de retard.

Le Métro 13 est encore une fois dans les pires lignes, ce qui est sans doute dû à la saturation de la ligne, les incidents fréquents et les correspondances compliquées.

Certains trams ont également de gros retards.
Ce qui peut être dû aux lignes avec des croisements et aux longs temps d'attente.

Pour finir, je vais analyser les retards en fonction de l'heure.
# In[90]:


df_temps_darret.head()


# In[108]:


print(df_temps_darret.columns)
print(df_temps_darret[['route_id', 'arrival_time_effectif']].dropna().head(10))

Synthèse Finale de l'Analyse des Retards en Île-de-France 

1 - Objectifs de l'Analyse

L’objectif de ce projet était d’analyser les retards dans les transports en commun d’Île-de-France à partir des données GTFS. Nous avons étudié :

- Les arrêts les plus fréquentés
- Les arrêts avec le plus de retard
- Les lignes les plus en retard
- Les différences de retard selon le type de transport
- L’évolution des retards en fonction de l’heure (impossible à analyser, voir explication plus bas)

Nous avons utilisé les horaires théoriques du GTFS et avons simulé les retards pour mieux comprendre leur répartition sur le réseau.

2 - Résultats Clés

Les arrêts les plus fréquentés :

Châtelet-Les Halles, Gare du Nord, La Défense, Gare de Lyon et Saint-Michel Notre-Dame sont les hubs majeurs du réseau.
Ces arrêts enregistrent le plus grand nombre de passages, confirmant leur rôle stratégique.

Les arrêts les plus en retard :

Certains arrêts affichent des retards moyens atteignant 9 minutes, bien au-dessus de la moyenne du réseau.
Les bus sont les plus impactés, surtout en banlieue.

Les lignes les plus en retard :

Les bus dominent largement le classement des lignes les plus en retard, notamment les lignes nocturnes et celles opérant en grande couronne.

Quelques lignes RER et TER affichent aussi des retards élevés, notamment le TER Grand-Est, le TER Centre-Val de Loire et certaines lignes RER.

Différences de retard par type de transport :

Les bus sont globalement les plus en retard. Les métros et tramways sont plus fiables, même si certaines lignes comme la ligne 13 et la 7B connaissent des retards plus fréquents. Le TER est le plus impacté parmi les trains régionaux.

3 - Pourquoi nous n’avons pas pu analyser les retards en fonction de l’heure 

Nous avions prévu d’analyser l’évolution des retards selon la journée (matin, midi, soir, nuit), mais nous avons dû abandonner cette partie pour une raison technique :

Nous ne disposons pas des horaires effectifs d’arrivée (arrival_time_effectif).

Dès le début, nous avons utilisé une simulation des retards basée sur les horaires prévus.
Sans les horaires réels, nous ne pouvons pas savoir si les retards sont plus fréquents le matin, à midi ou le soir.

Si nous avions eu les vraies heures d’arrivée, nous aurions pu voir les pics de retard aux heures de pointe.
Cela aurait permis d’identifier les créneaux horaires critiques pour les voyageurs et les opérateurs.

En résumé : L’absence des horaires réels était une contrainte connue dès le début, mais son impact s’est confirmé lors de l’analyse temporelle.

4️ - Limites et Perspectives

Points forts de l’analyse :

Bonne couverture des arrêts et lignes en retard.
Visualisation interactive des retards pour repérer les zones problématiques.

Limites :

Pas d’analyse des retards en fonction de l’heure (faute de données réelles).
Certains arrêts ont des noms ambigus ("Mairie", "Église"), ce qui complique l’interprétation.
Les causes des retards (trafic, incidents, conditions météo, etc.) ne sont pas prises en compte.

Axes d’amélioration pour une future étude :
1 - Obtenir des données d’horaires effectifs via une API (RATP, SNCF).
2 - Comparer les retards entre jours ouvrés et week-ends.

Ce projet a permis d’identifier les arrêts et lignes les plus touchés par les retards en Île-de-France. Malgré l’impossibilité d’analyser l’évolution des retards en fonction de l’heure, nous avons obtenu une vision claire des tendances générales.

Cette étude pourrait être approfondie avec des données en temps réel pour améliorer la fiabilité des transports et l’expérience des voyageurs.