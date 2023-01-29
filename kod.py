import csv
import pandas as pd
import plotly.express as px
import geopandas as gpd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

#odczyt pliku csv
df = pd.read_csv(r'Hotel_Reviews.csv')
df

#nowa kolumna Country
df['Country']=''
for i in range(0,len(df)):
  df['Country'][i]=df['Hotel_Address'][i].split()[-1]
  if df['Country'][i] == 'Kingdom':
     df['Country'].values[i] = df['Country'].values[i].replace('Kingdom','UK')
     
#ilosc hoteli wykres
ilosc_hoteli=df['Country'].value_counts()
ilosc_hoteli=pd.DataFrame(ilosc_hoteli).reset_index()
ilosc_hoteli.columns=['Country','count']
ilosc_hoteli
#sns.countplot(df2, x='Country')
sns.barplot(data=ilosc_hoteli, y='Country',x='count', palette=sns.cubehelix_palette(), orient='h')

#srednie oceny w podziale na Country
oceny_srednio=df.groupby('Country').Average_Score.mean()
oceny_srednio=pd.DataFrame(oceny_srednio).reset_index()
oceny_srednio.columns=['Country','srednia_mean']
oceny_srednio
#sns.barplot(data=oceny_srednio,x="Country",y='srednia_mean', palette="ch:.25")

#wspolrzedne hoteli
mapa=df[['Hotel_Name','lat','lng']].drop_duplicates()
mapa

#mapa
fig = px.scatter_mapbox(mapa, lat="lat", lon="lng", hover_name="Hotel_Name",
                        color_discrete_sequence=["fuchsia"], zoom=3.5, height=300)
fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        }
      ])
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

#mapa2
fig = px.scatter_mapbox(mapa, lat="lat", lon="lng", hover_name="Hotel_Name",
                        color_discrete_sequence=["fuchsia"], zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

#hotele hiszpania
spain=df.loc[df['Country'] == 'Spain']
spain

sp_dup=spain.drop_duplicates('Hotel_Name')
sp_dup

sp_av=spain.groupby(spain['Hotel_Name']).mean('Average_Score')
sp_av=sp_av[['Average_Score']]

sp_av_round=sp_av[['Average_Score']].round(0).groupby(['Average_Score'])['Average_Score'].count()

sp_av_round

#wykres średnie oceny hiszpanskich hoteli %
colors = sns.cubehelix_palette()
sp_av_round.plot.pie(autopct="%.1f%%",  colors=colors);

#srednia ocena hiszpanskich hoteli/ funkcja
sl=[]
for i in range(len(sp_dup)):
  sl.append(sp_dup['Average_Score'].values[i])

def obliczenie_sredniej_z_tablicy(tablica_liczb):
  suma = 0
  for element in tablica_liczb:
    suma+=element
  srednia = suma/len(tablica_liczb)
  return srednia
obliczenie_sredniej_z_tablicy(sl)

#ile hoteli powyzej sredniej
spain_ponad_srednia=sp_av.loc[sp_av['Average_Score'] > 8.49]
spain_ponad_srednia[['Average_Score']].count() 

#max srednia
spain[['Average_Score']].max()

#najlepsze hotele
best=spain.loc[spain['Average_Score'] ==9.6]
best.drop_duplicates('Hotel_Name')

#ilosc poz i neg komentarzy najlepszych hoteli
camper=best.where(best['Hotel_Name']=='Hotel Casa Camper')
neg_camper=camper['Negative_Review'].dropna()
neg_camper=camper['Negative_Review'].count()
noneg_camper=camper['Negative_Review'].where(camper['Negative_Review'] == 'No Negative').count()
n_c=neg_camper-noneg_camper

pos_camper=camper['Positive_Review'].dropna().count()
pos=camper['Negative_Review'].where(camper['Negative_Review'] == 'No Positive').count()
p_c=pos_camper-pos


serras=best.where(best['Hotel_Name']=='Hotel The Serras')
neg_serras=serras['Negative_Review'].dropna()
neg_serras=serras['Negative_Review'].count()
noneg_serras=serras['Negative_Review'].where(serras['Negative_Review'] == 'No Negative').count()
n_s=neg_serras-noneg_serras

pos_serras=serras['Positive_Review'].dropna().count()
pos=serras['Negative_Review'].where(serras['Negative_Review'] == 'No Positive').count()
p_s=pos_serras-pos


mimosa=best.where(best['Hotel_Name']=='H10 Casa Mimosa 4 Sup')
neg_mimosa=mimosa['Negative_Review'].dropna().count()
noneg_mimosa=mimosa['Negative_Review'].where(mimosa['Negative_Review'] == 'No Negative').count()
n_m=neg_mimosa-noneg_mimosa

pos_mimosa=mimosa['Positive_Review'].dropna().count()
pos=mimosa['Negative_Review'].where(mimosa['Negative_Review'] == 'No Positive').count()
p_m=pos_mimosa-pos


d = {'negative': [n_c, n_s, n_s], 'positive': [p_c,p_s, p_m]}

new= pd.DataFrame(data=d,index = ['camper', 'serras', 'mimosa'])
new

#wykres poz/neg komentarzt

  
X = new.index
positive = list(new.iloc[:,0])
negative = list(new.iloc[:,1])

  
X_axis = np.arange(len(X))
color1=sns.cubehelix_palette(1)
color2=sns.cubehelix_palette(1, gamma=3)

plt.bar(X_axis - 0.2,positive, 0.4, label = 'positive', color=color1)
plt.bar(X_axis + 0.2,negative, 0.4, label = 'negative',color=color2)

#colors = sns.color_palette('ch:.25')
#sp_av_round.plot.pie(autopct="%.1f%%",  colors=colors);
plt.color=sns.cubehelix_palette(2)
plt.xticks(X_axis, X)
plt.xlabel("Hotel")
plt.ylabel("Number of Reviews")
plt.title("Number of Reviews in each group of 3 best hotels in spain")
plt.legend()
plt.show()

#dataframe
new=new.reset_index()
new=pd.DataFrame(new)
new.columns=['Hotel','positive','negative']
new


#wykresy narodowosci gosci hotelowych(po recenzjach)
camper['Reviewer_Nationality'].value_counts()[:5].plot.pie(autopct="%.1f%%",colors = sns.cubehelix_palette(), title='Reviewer Nationality-CAMPER', ylabel='')
serras['Reviewer_Nationality'].value_counts()[:5].plot.pie(autopct="%.1f%%",colors = sns.cubehelix_palette(), title='Reviewer Nationality-SERRAS', ylabel='')
mimosa['Reviewer_Nationality'].value_counts()[:5].plot.pie(autopct="%.1f%%",colors = sns.cubehelix_palette(),title='Reviewer Nationality-MIMOSA', ylabel='')






#neg opinie camper
dni_c=camper[['Hotel_Name','Average_Score','Negative_Review','Positive_Review','days_since_review']].dropna().sort_values(by=['days_since_review'], ascending=False)
Cam_lastneg=dni_c['Negative_Review'].value_counts().head(15)
Cam_lastneg=pd.DataFrame(Cam_lastneg)
#Last_noneg=dni_c.where(dni_c['Negative_Review']=='No Negative').count()
#data['column_name'].value_counts()[value]
Cam_lastneg=pd.DataFrame(Cam_lastneg).reset_index()

Cam_lastneg.columns=['Review','Ile']

#list(Cam_lastneg.columns)
Cam_lastneg

#wykres
#Cam_lastneg.plot(x ='Review', y='Ile', kind='bar')
sns.barplot(data=Cam_lastneg.head(5), x='Review',y='Ile', palette=sns.cubehelix_palette())

#usuniete bledne wartosci
Camper_noneg=dni_c=camper[['Hotel_Name','Average_Score','Negative_Review','Positive_Review','days_since_review']].loc[(camper['Negative_Review'] != 'No Negative')&(camper['Negative_Review'] != ' Nothing')&(camper['Negative_Review'] != ' nothing')
&(camper['Negative_Review'] != 'Nothing')&(camper['Negative_Review'] != '  None ')&(camper['Negative_Review'] != 'nothing')].dropna()

Camper_noneg


#nowy wykres neg/poz opinie dla campera
#Camper_noneg.value_counts(['Negative_Review'])
neg_new=Camper_noneg['Negative_Review'].count()
pos_new=Camper_noneg['Positive_Review'].count()
pn_new = {'negative': [neg_new], 'positive': [pos_new]}
new2= pd.DataFrame(data=pn_new,index = ['camper'])
new2=pd.DataFrame(new2).reset_index()
new2.columns=['name','negative','positive']
new2


  
X = new2.index
positive = list(new2.iloc[:,1])
negative =list(new2.iloc[:,2])

  
X_axis = np.arange(len(X))
color1=sns.cubehelix_palette(1)
color2=sns.cubehelix_palette(1, gamma=3)
plt.bar(X_axis - 0.2,positive, 0.4, label = 'positive', color=color1)
plt.bar(X_axis + 0.2,negative, 0.4, label = 'negative',color=color2)


  
plt.xticks(X_axis, X)
plt.ylim([0,200])
plt.xlabel("Hotel")
plt.ylabel("Number of Reviews")
plt.title("Number of Reviews in each group of 3 best hotels in spain")
plt.legend()
plt.show()
