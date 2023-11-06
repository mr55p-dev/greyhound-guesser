# Greyhound guesser

Project for predicting the winner of a dog race, using an ML method.

I want to be able to accurately predict the probability of a given dog winning a given race. Placing in the top three is also relevant, but any position lower than 3 is considered a loss. The output space is three distinct:

1. The dog will place first
2. The dog will place in the top three. Note the loss function must be constructed to penalise predicting this outcome when the dog comes first, otherwise this will always be favoured to picking outcome 1.
3. The dog does not win.

I will just focus on the reduced task of predicting the winner of each race, given the six dogs and some statistics about them.

## Features:

### Global:

- **Race length**
  between 0 and 1000m

### For each dog:

- **Odds**
  between 0 and 1
- **Recent distance**
  between 0 and 1000m
- **Recent finish**
  between 1 and 6

## Other info

- **Data**: training data is taken from [kaggle](https://www.kaggle.com/datasets/davidregan/greyhound-racing-uk-predict-finish-position).
