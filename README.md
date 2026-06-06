# Artist Fame vs. Audio Features: Correlation with Track Popularity

DSC 80 Project @ UCSD

By: Matthew Ramos & Mayur Nookala

# Introduction

## General Introduction

Music popularity is influenced by many different factors. Some songs become popular because they are catchy, trendy, and easy to dance to. However, popularity may also depend on the artist behind the song. A song by a famous artist may receive more attention simply because the artist already has a large audience.

For this project, we worked with a Spotify music dataset that includes information about songs, artists, and audio features. The dataset contains information such as track popularity, danceability, energy, valence, acousticness, and whether or not a song is explicit. It also includes artist-level information such as artist followers and artist popularity. These columns allow us to compare two possible explanations for track popularity: the qualities of the song itself and the fame of the artist.

The main question we are interested in is:

**Does artist fame matter more than audio features when predicting track popularity?**

This question is important because it helps us understand whether a song’s success is mostly connected to how the song sounds, or whether it is strongly influenced by the artist’s existing popularity. In other words, we want to see if audio features like danceability, energy, and valence are strong predictors of track popularity, or if artist fame gives songs a bigger advantage.

To answer this question, we used data cleaning, exploratory data analysis, hypothesis testing, and machine learning. First, we will explore patterns between artist fame and track popularity. Then, we will compare track popularity between high-fame and low-fame artists. Finally, we will build prediction models to see whether artist-related features or audio features are more useful for predicting track popularity.

## Introduction of Columns

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

After cleaning and merging the datasets, our final dataset contains 92,619 rows and 20 columns. Overall, these columns help us compare artist fame with audio features. By looking at both types of information, we can better understand which factors are most connected to track popularity. 

# Data Cleaning and Exploratory Data Analysis

## Data Cleaning

In this section, we cleaned the Spotify track and artist datasets so they can be used for our analysis. Since our project focuses on whether artist fame matters more than audio features when predicting track popularity, we keep columns related to track popularity, artist popularity, artist followers, genre, and audio features.

We also renamed columns so that track popularity and artist popularity are clearly different. Then, we merged the track dataset with the artist dataset so that each row contains information about a song and the artist who made it. Finally, we removed rows with missing values in important columns and create new columns that will help with our analysis, such as `log_followers` and `fame_group`.

### Head of Cleaned DataFrame

<div style="overflow-x: auto; max-width: 100%;">

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>track_name</th>
      <th>artist_name</th>
      <th>track_popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>track_genre</th>
      <th>artist_popularity</th>
      <th>artist_followers</th>
      <th>log_followers</th>
      <th>duration_minutes</th>
      <th>fame_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Comedy</td>
      <td>Gen Hoshino</td>
      <td>73</td>
      <td>230666</td>
      <td>False</td>
      <td>0.676</td>
      <td>0.4610</td>
      <td>-6.746</td>
      <td>0.1430</td>
      <td>0.0322</td>
      <td>0.000001</td>
      <td>0.3580</td>
      <td>0.715</td>
      <td>87.917</td>
      <td>acoustic</td>
      <td>66.0</td>
      <td>852637.0</td>
      <td>13.656090</td>
      <td>3.844433</td>
      <td>High Fame</td>
    </tr>
    <tr>
      <td>Ghost - Acoustic</td>
      <td>Ben Woodward</td>
      <td>55</td>
      <td>149610</td>
      <td>False</td>
      <td>0.420</td>
      <td>0.1660</td>
      <td>-17.235</td>
      <td>0.0763</td>
      <td>0.9240</td>
      <td>0.000006</td>
      <td>0.1010</td>
      <td>0.267</td>
      <td>77.489</td>
      <td>acoustic</td>
      <td>53.0</td>
      <td>11874.0</td>
      <td>9.382191</td>
      <td>2.493500</td>
      <td>Low Fame</td>
    </tr>
    <tr>
      <td>To Begin Again</td>
      <td>Ingrid Michaelson</td>
      <td>57</td>
      <td>210826</td>
      <td>False</td>
      <td>0.438</td>
      <td>0.3590</td>
      <td>-9.734</td>
      <td>0.0557</td>
      <td>0.2100</td>
      <td>0.000000</td>
      <td>0.1170</td>
      <td>0.120</td>
      <td>76.332</td>
      <td>acoustic</td>
      <td>68.0</td>
      <td>722496.0</td>
      <td>13.490469</td>
      <td>3.513767</td>
      <td>High Fame</td>
    </tr>
    <tr>
      <td>Can't Help Falling In Love</td>
      <td>Kina Grannis</td>
      <td>71</td>
      <td>201933</td>
      <td>False</td>
      <td>0.266</td>
      <td>0.0596</td>
      <td>-18.515</td>
      <td>0.0363</td>
      <td>0.9050</td>
      <td>0.000071</td>
      <td>0.1320</td>
      <td>0.143</td>
      <td>181.740</td>
      <td>acoustic</td>
      <td>71.0</td>
      <td>438860.0</td>
      <td>12.991938</td>
      <td>3.365550</td>
      <td>High Fame</td>
    </tr>
    <tr>
      <td>I'm Yours</td>
      <td>Jason Mraz</td>
      <td>80</td>
      <td>242946</td>
      <td>False</td>
      <td>0.703</td>
      <td>0.4440</td>
      <td>-9.331</td>
      <td>0.0417</td>
      <td>0.5590</td>
      <td>0.000000</td>
      <td>0.0973</td>
      <td>0.712</td>
      <td>150.960</td>
      <td>acoustic</td>
      <td>79.0</td>
      <td>5961911.0</td>
      <td>15.600902</td>
      <td>4.049100</td>
      <td>High Fame</td>
    </tr>
  </tbody>
</table>

</div>

## Univariate Analysis

For our univariate analysis, we first looked at the distribution of `track_popularity`, which is the variable we are trying to understand and eventually predict. We also looked at the distribution of `log_followers` because artist followers is an important part of how we measure artist fame. Using the log of followers makes the distribution easier to interpret because raw follower counts can be extremely large for famous artists.

<iframe src="assets/css/track-popularity-distribution.html" width="100%" height="500" frameborder="0"></iframe>

<iframe src="assets/css/log-followers-distribution.html" width="100%" height="500" frameborder="0"></iframe>

## Bivariate Analysis

For our bivariate analysis, we looked at the relationship between artist fame and track popularity. We first looked at the correlation between different predictor variables with track popularity. It is found that `artist_popuarity` has the highest correlation to `track_popularity`, with a correlation value of 0.22.

<iframe src="assets/css/correlation_plot.html" width="100%" height="500" frameborder="0"></iframe>

We also used scatter plots to help compare `log_followers` and `artist_popularity` against `track_popularity`. These plots help us see whether songs by more famous artists tend to have higher popularity scores.

<iframe src="assets/css/followers-vs-track-popularity.html" width="100%" height="500" frameborder="0"></iframe>

<iframe src="assets/css/artist-popularity-vs-track-popularity.html" width="100%" height="500" frameborder="0"></iframe>

## Interesting Aggregates

For our interesting aggregates, we grouped songs by `fame_group` and compared the average track popularity of high-fame and low-fame artists. If high-fame artists have higher average track popularity, then this suggests that artist fame may be related to song popularity.

We also grouped songs by genre to see whether certain genres tend to have higher average popularity. This is useful because genre may also affect track popularity and could be useful in our prediction model.

### Average Popularity by Fame Group

| Fame Group | Avg. Track Popularity | Avg. Artist Popularity | Avg. Log Followers | Number of Tracks |
|---|---:|---:|---:|---:|
| High Fame | 39.35 | 68.10 | 13.79 | 46,305 |
| Low Fame | 31.57 | 37.52 | 8.74 | 46,314 |

This table shows that songs by high-fame artists had higher average track popularity than songs by low-fame artists. High-fame artists also had much higher artist popularity and follower counts, which supports the idea that artist fame may be related to track popularity.

### Top Genres by Average Track Popularity

| Genre | Avg. Track Popularity | Avg. Artist Popularity | Number of Tracks |
|---|---:|---:|---:|
| k-pop | 59.49 | 68.91 | 849 |
| pop-film | 59.12 | 62.59 | 893 |
| chill | 55.20 | 51.91 | 684 |
| pop | 54.38 | 71.41 | 760 |
| sad | 52.38 | 52.11 | 794 |
| indian | 50.51 | 57.15 | 728 |
| grunge | 50.14 | 65.54 | 974 |
| anime | 49.60 | 53.15 | 1,057 |
| emo | 48.64 | 62.27 | 858 |
| sertanejo | 47.77 | 63.05 | 857 |

This table shows the genres with the highest average track popularity in the cleaned dataset. Genres like k-pop, pop-film, chill, and pop had the highest average popularity scores, which suggests that genre may also be related to how popular a track becomes.

# Assessment of Missingness

## NMAR Analysis

We consider `artist_followers` to be potentially **NMAR** in the dataset. We are missing the followers' count of some artists, and we suppose that this may happen because because those artists may be less known or harder to match between datasets. In other words, the value of `artist_followers` is missing because it is very small — the lesser known the artist, the higher the chance of missing data.

However, the case might turn into MAR if there was additional information on how the dataset was created. If we knew how the name of the artist was misspelled or how his or her name did not match, then it could explain the absence of followers in the dataset.

## Missingness Dependency

For the missingness dependency test, we analyze whether missingness in `artist_followers` depends on other columns.
For each test:

**Null Hypothesis:** The missingness of `artist_followers` does not depend on the comparison column.

**Alternative Hypothesis:** The missingness of `artist_followers` does depend on the comparison column.

**Test Statistic:** Absolute difference in mean comparison-column value between rows where `artist_followers` is missing and rows where `artist_followers` is not missing.

**Significance Level:** 0.05

### Missingness Dependency Results

| Missing Column | Comparison Column | Observed Statistic | p-value |
|---|---|---:|---:|
| artist_followers | track_popularity | 2.710526 | < 0.001 |
| artist_followers | energy | 0.112925 | < 0.001 |
| artist_followers | valence | 0.064566 | < 0.001 |
| artist_followers | acousticness | 0.114472 | < 0.001 |
| artist_followers | tempo | 6.297681 | < 0.001 |
| artist_followers | duration_ms | 29601.760962 | < 0.001 |
| artist_followers | danceability | 0.003517 | 0.204 |

The missingness permutation tests suggest that missingness in `artist_followers` depends on `track_popularity`, but does not appear to depend on `danceability`.

For `track_popularity`, the observed statistic was about 2.71 and the simulated p-value was less than 0.001. Since this p-value is below our significance level of 0.05, we reject the null hypothesis. This means there is evidence that missingness in `artist_followers` depends on `track_popularity`.

For `danceability`, the observed statistic was about 0.0035 and the p-value was 0.204. Since this p-value is above 0.05, we fail to reject the null hypothesis. This means we do not have evidence that missingness in `artist_followers` depends on `danceability`.

<iframe src="assets/css/missingness-track-popularity.html" width="100%" height="500" frameborder="0"></iframe>

# Hypothesis Testing

For our hypothesis test, we wanted to determine whether songs by high-fame artists have higher track popularity than songs by low-fame artists.

We separated artists into two groups using the median value of `log_followers``. Artists above the median were labeled as high-fame artists, while artists at or below the median were labeled as low-fame artists.

**Null Hypothesis:** There is no difference in average track popularity between songs by high-fame artists and songs by low-fame artists.

**Alternative Hypothesis:** Songs by more famous artists have a higher average track popularity than songs by low-fame artists.

**Test Statistic:** Difference in mean track popularity between high-fame artists and low-fame artists.

**Significance level:** 0.05

<iframe src="assets/css/hypothesis-test.html" width="100%" height="500" frameborder="0"></iframe>

The observed difference in mean track popularity was about 7.78 points, meaning high-fame artists had tracks that were about 7.78 popularity points higher on average than low-fame artists. In the permutation test, the shuffled differences were centered near 0, while the observed difference was far to the right of the null distribution. The p-value was < 0.001, so we reject the null hypothesis, suggesting that tracks by high-fame artists tend to have higher popularity than tracks by low-fame artists.

# Prediction Problem

Our prediction problem is:

**"Can we predict a song's Spotify track popularity using artist fame and audio features?"** which makes it a regression problem.

The response variable is `track_popularity.`

If artist fame helps predict track popularity better than audio features alone, then fame is an important part of the story.

We used **RMSE** as the main evaluation metric, as it measures how far the predicted popularity scores are from the real popularity scores. The lower the RMSE, the better.

At the time of prediction, we would know features like artist followers, artist popularity, audio features, duration, and whether the song is explicit. We should not use any feature that directly leaks the track popularity value.

# Baseline Model

For the baseline model, we used a linear regression model. It uses four features: three quantitative features and one nominal feature.

- `log_followers` — quantitative (a log transform of the original `artist_followers` column, used because raw follower counts are heavily right-skewed)
- `danceability` — quantitative
- `energy` — quantitative
- `explicit` — nominal

The three quantitative columns were standardized with `StandardScaler`, and the nominal `explicit` column was one-hot encoded. All transformations and model training were implemented in a single sklearn Pipeline.

The baseline model had an RMSE of about 21.21 and an R² value of about 0.037. This means the model’s predictions were typically off by around 21 popularity points, and the model only explained about 3.7% of the variation in track popularity. Because of this, I would not consider the baseline model very strong. However, it is still useful as a starting point when being used to compare to the final model.

# Final Model

For the final model, we used a DecisionTreeRegressor because the relationship between artist fame, audio features, and track popularity may not be perfectly linear, and a decision tree can capture non-linear patterns better than a simple linear regression model.

We tuned two hyperparameters using GridSearchCV:

* `max_depth`: how deep the decision tree is allowed to grow
* `min_samples_leaf`: the minimum number of samples required in each leaf

The final model also adds engineered features:

* `energy_danceability`: combines energy and danceability, since songs that are both energetic and danceable may behave differently from songs that are only high in one of those features.
* `high_energy`: marks whether a song has above-median energy, which helps the model separate higher-energy songs from lower-energy songs.

The final model also uses existing features such as `log_followers`, `artist_popularity`, audio features, `duration_minutes`, `explicit`, and `track_genre`. All feature engineering, preprocessing, hyperparameter tuning, and model training are done through sklearn tools.

The final Decision Tree model performed better than the baseline linear regression model. The baseline model had an RMSE of about 21.21 and an R² of about 0.037, while the final model had an RMSE of about 18.74 and an R² of about 0.248.

The final model reduced the RMSE by about 2.47 popularity points compared to the baseline model. The R² value also increased, meaning the final model explained more of the variation in track popularity.

This improvement likely happened because the final model used more useful features, included engineered features like `energy_danceability` and `high_energy`, and used a Decision Tree model that can capture non-linear relationships better than a simple Linear Regression model.

# Fairness Analysis

For the fairness analysis, we check whether the final model performs worse for high-fame artists than for low-fame artists.

**Group X:** High-fame artists  
**Group Y:** Low-fame artists

**Evaluation Metric:** RMSE

**Null Hypothesis:** The model is fair. Its RMSE for high-fame and low-fame artists is roughly the same, and any difference is due to random chance.

**Alternative Hypothesis:** The model is unfair. Its RMSE for high-fame artists is higher than its RMSE for low-fame artists.

**Test Statistic:** RMSE for high-fame artists minus RMSE for low-fame artists.

**Significance Level:** 0.05

<iframe src="assets/css/fairness-test.html" width="100%" height="500" frameborder="0"></iframe>

The RMSE for low-fame artists was about 15.34, while the RMSE for high-fame artists was about 21.60. Since our test statistic was RMSE for high-fame artists minus RMSE for low-fame artists, the observed statistic was about 6.26.

The p-value was < 0.001 in our simulation, meaning none of the shuffled trials produced an RMSE difference as large as the observed one. That means we reject the null hypothesis.

This suggests that the final model performs worse for high-fame artists than for low-fame artists. However, this does not prove that fame causes the model to perform worse; it only shows that there is a noticeable difference in model error between the two groups in our test set.