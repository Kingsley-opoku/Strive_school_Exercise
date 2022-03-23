import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

import requests
from bs4 import BeautifulSoup

URL = ["https://www.imdb.com/search/title/?genres=action&explore=title_type,genres&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e0da8c98-35e8-4ebd-8e86-e7d39c92730c&pf_rd_r=NCWR107JR3D070SM7GBG&pf_rd_s=center-2&pf_rd_t=15051&pf_rd_i=genre&ref_=ft_gnr_pr2_i_2",
                "https://www.imdb.com/search/title/?genres=action&start=51&explore=title_type,genres&ref_=adv_nxt"]
  


# Creating the lists that we want to 
description =[]
duration=[]
release_date=[]
movie_director_stars=[]
movie_year=[]
director_sep = []
stars_sep = []
rating = []
genre=[]

votes=[]


movie_name_100=[]

for url in range(0,2):
    req = requests.get(URL[url])
    soup = BeautifulSoup(req.text, 'html.parser')
    for container in soup.find_all("h3" ,class_="lister-item-header"):
        movie_name_100.append(container.text.replace('\n', '').strip(')').split('(')[0].split('.')[1])
        movie_year.append(container.text.replace('\n', '').strip(')').split('(')[1].split('–')[0]) 
    
    # for release in soup.find_all('span', class_="runtime"):
    #     duration.append(release.text) 

    
    for dist in soup.select('.text-muted+ .text-muted , .ratings-bar+ .text-muted'):
        description.append(dist.text.replace('\n', ''))

    for director in soup.find_all('p', class_ =''):
        director_sep.append(director.text.replace("\n", "").split('|')[0].split(':')[1].split(',')[0])
        stars_sep.append(director.text.replace("\n",'').split('Stars:')[1])

    # for years in soup.find_all('span', class_ ='lister-item-year'):
    #     movie_years.append(years.text)
    
    
    for elements in soup.select('.lister-item-content'):
        extract_rating=pd.Series(elements.text.replace('\n', '')).str.extract(r'(\d\.\d)Rate this')
        rating.append(float(extract_rating.squeeze()))
        dura=pd.Series(elements.text.replace('\n','')).str.extract(r'(\d+) min').squeeze()
        duration.append(dura)


    for genres in soup.find_all('span', class_ = 'genre'):
        genre.append(genres.text.replace('\n','').strip())

    for vote in soup.find_all('div', class_='lister-item-content'):
        extract_vote=pd.Series(vote.text.replace('\n','')).str.extract(r'Votes:(\d+,?\d+)').squeeze()
        
        if isinstance(extract_vote, str):
            votes.append(int(extract_vote.replace(',','')))
        else:
            votes.append(extract_vote)


df=pd.DataFrame({'Title':  movie_name_100, 'Description':description, 'Release':movie_year , 
                        'Director': director_sep, 
                        'Rating': rating, 'Duration':duration, 'Votes': votes,
                        'Genre':genre, 'Stars': stars_sep, })



st.title('Imdb Best Action Movies')
st.write('Visualising first 5 row of the data')
st.write(df[:5])
#rating_graph=df["Rating"].value_counts().nlargest(10).plot(kind="bar", title="Top 10 rating range of movies ", figsize=(10,8))
st.subheader('Top 10 rating range of movies with the number of movies')

st.bar_chart(data=df['Rating'].value_counts().nlargest(10)
                 )


st.subheader('Number of movie released by years')
st.bar_chart(df["Release"].value_counts().nlargest(10))


st.subheader('Number of movies by each director')

directors=df["Director"].value_counts().nlargest(10)

st.bar_chart(directors)


#st.area_chart(df['Director'].value_counts().nlargest(10))

