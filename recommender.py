import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def preprocessing(data):
    filter_movies = data.groupby('movieId').filter(lambda x : len(x) >= 5)
    filter_users = filter_movies.groupby('userId').filter(lambda x : len(x) >= 5)

    return filter_users

def prediction_with_weighted_average(test_df, pearson_corr, knn):
    
    user_predictions = {}
    total_users = len(test_df)
    total_movies = len(test_df.columns)
    predictions_df = pd.DataFrame(index = test_df.index, columns = test_df.columns)

    for user in range (0, total_users):
        
        predicted_ratings = {}
        userID = test_df.index[user]

        for movie in range (0, total_movies):
            
            movieID = test_df.columns[movie]

            if not np.isnan(test_df._get_value(userID, movieID)):                   # if the movie has a rank to compare to
                
                sim_movies = pearson_corr[movieID]
                sim_movies.dropna(inplace = True)                                   # Drop nan values
                sim_movies.drop(sim_movies[sim_movies < 0 ].index , inplace = True) # Drop negative values
                sim_movies = sim_movies.sort_values(ascending = False)              # Sort
                
            else:
                continue
            if sim_movies.empty:
                predicted_ratings[movieID] = np.nan
                predictions_df.at[userID, movieID] = np.nan
                continue
            
            top_sim_dic = {}
            nn_count = 0
            
            for i in range (0, len(sim_movies)):

                if nn_count == knn:
                    break
                if sim_movies.index[i] == movieID:
                    continue

                rating = test_df._get_value(userID, sim_movies.index[i])
                if np.isnan(rating):
                    continue
                else:
                    top_sim_dic[sim_movies.index[i]] = rating
                    nn_count += 1

            top_sim = pd.Series(top_sim_dic, dtype="float64")

            if top_sim.empty:
                predicted_ratings[movieID] = np.nan
                predictions_df.at[userID, movieID] = np.nan
                continue

            predicted_rating = 0
            correlation_values_sum = 0

            for j in range (0, len(top_sim)):
                rating = top_sim.get(top_sim.index[j])
                corr_value = sim_movies.get(top_sim.index[j])
                correlation_values_sum += corr_value
                predicted_rating += rating * corr_value
             
            predicted_rating = predicted_rating / correlation_values_sum
            predicted_rating = round(predicted_rating * 2) / 2
            predicted_ratings[movieID] = predicted_rating
            predictions_df.at[userID, movieID] = predicted_rating
        
        user_predictions[userID] = predicted_ratings

    predictions_df.dropna(axis=1, how = "all", inplace = True)
    return predictions_df

def prediction_with_average(test_df, pearson_corr, knn):
    
    user_predictions = {}
    total_users = len(test_df)
    total_movies = len(test_df.columns)
    predictions_df = pd.DataFrame(index = test_df.index, columns = test_df.columns)

    for user in range (0, total_users):
        
        predicted_ratings = {}
        userID = test_df.index[user]

        for movie in range (0, total_movies):
            
            movieID = test_df.columns[movie]

            if not np.isnan(test_df._get_value(userID, movieID)):                   # if the movie has a rank to compare to
                
                sim_movies = pearson_corr[movieID]
                sim_movies.dropna(inplace = True)                                   # Drop nan values
                sim_movies.drop(sim_movies[sim_movies < 0 ].index , inplace = True) # Drop negative values
                sim_movies = sim_movies.sort_values(ascending = False)              # Sort
                
            else:
                continue
            if sim_movies.empty:
                predicted_ratings[movieID] = np.nan
                predictions_df.at[userID, movieID] = np.nan
                continue
            
            top_sim_dic = {}
            nn_count = 0
            
            for i in range (0, len(sim_movies)):

                if nn_count == knn:
                    break
                if sim_movies.index[i] == movieID:
                    continue

                rating = test_df._get_value(userID, sim_movies.index[i])
                if np.isnan(rating):
                    continue
                else:
                    top_sim_dic[sim_movies.index[i]] = rating
                    nn_count += 1

            top_sim = pd.Series(top_sim_dic, dtype="float64")

            if top_sim.empty:
                predicted_ratings[movieID] = np.nan
                predictions_df.at[userID, movieID] = np.nan
                continue

            predicted_rating = 0

            for j in range (0, len(top_sim)):
                rating = top_sim.get(top_sim.index[j])
                predicted_rating += rating
             
            predicted_rating = predicted_rating / nn_count
            predicted_rating = round(predicted_rating * 2) / 2
            predicted_ratings[movieID] = predicted_rating
            predictions_df.at[userID, movieID] = predicted_rating
        
        user_predictions[userID] = predicted_ratings

    predictions_df.dropna(axis=1, how = "all", inplace = True)
    return predictions_df

def prediction_with_weighted_common_user_average(train_df, test_df, pearson_corr, knn):
    
    user_predictions = {}
    total_users = len(test_df)
    total_movies = len(test_df.columns)
    predictions_df = pd.DataFrame(index = test_df.index, columns = test_df.columns)

    for user in range (0, total_users):
        
        predicted_ratings = {}
        userID = test_df.index[user]

        for movie in range (0, total_movies):
            
            movieID = test_df.columns[movie]

            if not np.isnan(test_df._get_value(userID, movieID)):                   # if the movie has a rank to compare to
                
                sim_movies = pearson_corr[movieID]
                sim_movies.dropna(inplace = True)                                   # Drop nan values
                sim_movies.drop(sim_movies[sim_movies < 0 ].index , inplace = True) # Drop negative values
                sim_movies = sim_movies.sort_values(ascending = False)              # Sort
                
            else:
                continue
            if sim_movies.empty:
                predicted_ratings[movieID] = np.nan
                predictions_df.at[userID, movieID] = np.nan
                continue
            
            top_sim_dic = {}
            nn_count = 0
            
            for sim_movie_i in range (0, len(sim_movies)):

                if nn_count == knn:
                    break
                if sim_movies.index[sim_movie_i] == movieID:
                    continue

                rating = test_df._get_value(userID, sim_movies.index[sim_movie_i])
                if np.isnan(rating):
                    continue
                else:
                    top_sim_dic[sim_movies.index[sim_movie_i]] = rating
                    nn_count += 1

            top_sim = pd.Series(top_sim_dic, dtype="float64")

            if top_sim.empty:
                predicted_ratings[movieID] = np.nan
                predictions_df.at[userID, movieID] = np.nan
                continue

            predicted_rating = 0
            correlation_values_sum = 0
            common_count = {}

            for top_sim_movie_i in range (0, len(top_sim)):

                count = 0
                for user_in_train in range (0, len(train_df)):
                    
                    if not (np.isnan(train_df._get_value(train_df.index[user_in_train], movieID)) or np.isnan(train_df._get_value(train_df.index[user_in_train], top_sim.index[top_sim_movie_i]))) :
                        count += 1
                        
                common_count[top_sim.index[top_sim_movie_i]] = count
                
            highest_common = pd.Series(common_count)
            highest_common = highest_common.sort_values(ascending = False)

            for top_common_movie_i in range (0, len(highest_common)):

                rating = top_sim.get(highest_common.index[top_common_movie_i])
                corr_value = sim_movies.get(top_sim.index[top_common_movie_i])
                correlation_values_sum += corr_value
                predicted_rating += rating * corr_value
             
            predicted_rating = predicted_rating / correlation_values_sum
            predicted_rating = round(predicted_rating * 2) / 2
            predicted_ratings[movieID] = predicted_rating
            predictions_df.at[userID, movieID] = predicted_rating

        user_predictions[userID] = predicted_ratings

    predictions_df.dropna(axis=1, how = "all", inplace = True)
    return predictions_df

def evaluation(test_df, prediction_df):

    mae_dic = {}
    for user in range(0, len(test_df)):

        if test_df.index[user] not in prediction_df.index:
            continue
        
        abs_error_total = 0
        count = 0
        for movie in range(0, len(test_df.columns)):
            
            if test_df.columns[movie] not in prediction_df.columns or np.isnan(test_df._get_value(test_df.index[user], test_df.columns[movie])):
                continue

            if not np.isnan(prediction_df._get_value(test_df.index[user], test_df.columns[movie])):
                abs_error_total += abs(prediction_df._get_value(test_df.index[user], test_df.columns[movie]) - test_df._get_value(test_df.index[user], test_df.columns[movie]))
                count += 1
        
        mae_dic[test_df.index[user]] = abs_error_total / count

    mae = pd.Series(mae_dic)

    total_relevant = 0
    total_retrieved = 0
    total_relevant_retrieved = 0
    
    for user in range(0, len(test_df)):

        if test_df.index[user] not in prediction_df.index:
            
            for movie in range(0, len(test_df.columns)):
                if test_df._get_value(test_df.index[user], test_df.columns[movie]) >= 3:
                    total_relevant += 1
        else:
            for movie in range(0, len(test_df.columns)):

                if test_df.columns[movie] not in prediction_df.columns:
                    if test_df._get_value(test_df.index[user], test_df.columns[movie]) >= 3:
                        total_relevant += 1
                else:
                    if test_df._get_value(test_df.index[user], test_df.columns[movie]) >= 3:
                        total_relevant += 1
                        
                        if prediction_df._get_value(test_df.index[user], test_df.columns[movie]) >= 3:
                            total_relevant_retrieved += 1
                            total_retrieved += 1
                        elif prediction_df._get_value(test_df.index[user], test_df.columns[movie]) < 3:
                            total_retrieved += 1
                    
                    elif test_df._get_value(test_df.index[user], test_df.columns[movie]) < 3:

                        if prediction_df._get_value(test_df.index[user], test_df.columns[movie]) >= 3:
                            total_retrieved += 1
                        

    precision = total_relevant_retrieved / total_retrieved
    recall = total_relevant_retrieved / total_relevant

    print("Mean average error is:", round(mae.mean(), 2))
    print("Precision is:", round(precision, 2))
    print("Recall is:", round(recall, 2))

    return

k = input("Enter K: ")
train_set_size = input("Enter percentage of training set as a float (e.g. 50% = 0.5): ")
k = int(k)
train_set_size = float(train_set_size)

df = pd.read_csv('ratings.csv')
df = pd.DataFrame(df, columns = ['userId', 'movieId', 'rating'])

# Filtering
df = preprocessing(df)

df = df.pivot(index = 'userId', columns = 'movieId', values = 'rating')
#print(df)
pearson_corr = df.corr()
#print(pearson_corr)

train_df, test_df = train_test_split(df, train_size = train_set_size, test_size = 0.1, shuffle = False)

print("Train set size = " + str(train_set_size) + " K = " + str(k))

predictions_df = prediction_with_average(test_df, pearson_corr, k)
print("Results of prediction with average")
evaluation(test_df, predictions_df)

predictions_df = prediction_with_weighted_average(test_df, pearson_corr, k)
print("Results of prediction with weighted average")
evaluation(test_df, predictions_df)

predictions_df = prediction_with_weighted_common_user_average(train_df, test_df, pearson_corr, k)
print("Results of prediction with weighted common user average")
evaluation(test_df, predictions_df)

#k_set = [3, 5, 7, 9, 12]

#for k in k_set:
#    print("K =", k)
#    predictions_df = prediction_with_average(test_df, pearson_corr, k)
#    print("Results of prediction with average")
#    evaluation(test_df, predictions_df)

#    predictions_df = prediction_with_weighted_average(test_df, pearson_corr, k)
#    print("Results of prediction with weighted average")
#    evaluation(test_df, predictions_df)

#    predictions_df = prediction_with_weighted_common_user_average(train_df, test_df, pearson_corr, k)
#    print("Results of prediction with weighted common user average")
#    evaluation(test_df, predictions_df)

#train_set = [0.5, 0.7, 0.9]
#k = 12
#for i in train_set:

#    print("train set size =", i)
#    train_df, test_df = train_test_split(df, train_size = i, test_size = 0.1, shuffle = False)
#    predictions_df = prediction_with_average(test_df, pearson_corr, k)
#    print("Results of prediction with average: ")
#    evaluation(test_df, predictions_df)

#    predictions_df = prediction_with_weighted_average(test_df, pearson_corr, k)
#    print("Results of prediction with weighted average: ")
#    evaluation(test_df, predictions_df)

#    predictions_df = prediction_with_weighted_common_user_average(train_df, test_df, pearson_corr, k)
#    print("Results of prediction with weighted common user average: ")
#    evaluation(test_df, predictions_df)