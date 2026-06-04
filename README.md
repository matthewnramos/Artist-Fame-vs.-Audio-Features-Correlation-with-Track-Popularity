# Artist Fame vs. Audio Features Correlation with Track Popularity

DSC 80 Project @ UCSD
By: Matthew Ramos & Mayur Nookala

## Introduction

### General Introduction

Music popularity is influenced by many different factors. Some songs become popular because they are catchy, energetic, easy to dance to, or fit current music trends. However, popularity may also depend on the artist behind the song. A song by a famous artist may receive more attention simply because the artist already has a large audience.

For this project, we are working with a Spotify music dataset that includes information about songs, artists, and audio features. The dataset contains information such as track popularity, danceability, energy, valence, acousticness, and whether or not a song is explicit. It also includes artist-level information such as artist followers and artist popularity. These columns allow us to compare two possible explanations for track popularity: the qualities of the song itself and the fame of the artist.

The main question we are interested in is:

**Does artist fame matter more than audio features when predicting track popularity?**

This question is important because it helps us understand whether a song’s success is mostly connected to how the song sounds, or whether it is strongly influenced by the artist’s existing popularity. In other words, we want to see if audio features like danceability, energy, and valence are strong predictors of track popularity, or if artist fame gives songs a bigger advantage.

To answer this question, we will use data cleaning, exploratory data analysis, hypothesis testing, and machine learning. First, we will explore patterns between artist fame and track popularity. Then, we will compare track popularity between high-fame and low-fame artists. Finally, we will build prediction models to see whether artist-related features or audio features are more useful for predicting track popularity.

### Introduction of Columns

The dataset includes a variety of columns that describe songs, artists, and audio features. Some of the most important columns for this project are listed below.

**track_name:** This column contains the name of each song.

**artist_name:** This column contains the name of the artist who made the song.

**track_popularity:** This column measures how popular a song is. This is the main column we are trying to understand and predict in this project.

**artist_followers:** This column shows how many followers an artist has. We use this as one way to measure artist fame.

**artist_popularity:** This column measures the popularity of the artist. This is another way to represent how well-known or successful an artist is.

**danceability:** This column measures how suitable a song is for dancing based on musical elements like tempo, rhythm, and beat strength.

**energy:** This column measures how intense or active a song feels. Songs with high energy usually feel faster, louder, or more powerful.

**valence:** This column measures how positive or happy a song sounds. Higher valence usually means the song sounds more cheerful, while lower valence means the song may sound more sad or serious.

**acousticness:** This column measures how acoustic a song is. A higher value means the song is more likely to use acoustic instruments instead of electronic or heavily produced sounds.

**explicit:** This column shows whether a song contains explicit content.

**genre:** This column describes the genre or category of the song. Genre is useful because popularity patterns may be different across different types of music.

Overall, these columns help us compare artist fame with audio features. By looking at both types of information, we can better understand which factors are most connected to track popularity.
