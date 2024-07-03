# Article Title

## Introduction

What problem are we trying to solve? Consider a scenario involving players who can compete against each other in pairs, resulting in one player winning and another losing. These players can be humans playing a game, such as chess. Alternatively, we might have a situation where people express their preferences by choosing one item (texts, videos, movies, pictures, songs) over another. In this case, the items themselves become the "players," with the chosen item "winning" and the other "losing".

Our goal is to be able to predict the outcomes of any two given "players" competing against each other. In other words, we want to be able to estimate the probability of any given player A to win against any other given player B, in case they are confronted to play agains each other. This question is relevant for pairs of players who have never played together as well as for those who have played together multiple times. For example, if players A and B have faced each other five times and player A won four out of those five games, we can estimate the probability of player A defeating player B as 0.8. However, we can make a much better estimate by examining player A's results against other players, as well as player B's results against other players. By considering a broader set of statistics, we can more accurately assess the strengths of both players and thus better predict the outcome when they play against each other.

## Data and Model
To begin, we define and introduce an appropriate data structure for our analysis. We propose using a square matrix $W$, where $W_{ij}$ represents the number of wins player $j$ has over player $j$. In this matrix, the $i$-th row contains all the wins of player $i$, and the $j$-th column contains all the losses of player $j$. We will refer to this as the wins matrix.
