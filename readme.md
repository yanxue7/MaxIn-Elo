# Learning to Identify Top Elo Ratings

We propose two algorithms MaxIn-Elo and MaxIn-mElo to solve the top players identification on the transitive and intransitive settings. 
All winning probability matrices of games are saved in file 'games/'.
You can install the required packages by: 
```
pip install -r requirements.txt
```

### Baselines

There are 5 baselines in mrandom.py, mDBGD.py, mRGUCB.py, mELOMLE.py, MaxInELO.py.

For mrandom.py, mDBGD.py, mRGUCB.py and MaxInELO.py, we set a parameter 'self.melo' to control using Elo or mElo to update ratings.



### Runs
##### Results of top-1 identification 
For the Elo model, you can tune the best parameters of top-1 performance on transitive games by running:
```
sh runelo.sh 
```
Then you can plot the results of top-1 on the Elo model by running:
```
python Elo_plot.py Max 0
```
All figures are save in file **finalplot/**. And the results of top-1 of Elo showed in figure 2 are obtained:

<img src="https://github.com/yanxue7/MaxIn-Elo/blob/main/src/figure2_of_elo.png" width="500"/>

For the mElo model, you can tune the best parameters of the top-1 performance on intransitive games by running:
```
sh runmelo.sh 
```
Then you can plot the results of top-1 on the mElo model by running:
```

python Elo_plot.py Max 1
```

The results of top-1 on mElo in Figure 3 are obtained:

<img src="https://github.com/yanxue7/MaxIn-Elo/blob/main/src/figure3_of_melo.png" width="500"/>

##### Results of top-k identification 

You can get the results of top-k identification of all baselines by running:
```
sh runelo.sh
python topk_plot.py
```

##### Comparison of different $\gamma$

You can get the results of different $\gamma$ of our MaxIn-Elo on transitive games by running:
```
python compare_gamma.py
```

##### Comparison of different dimension C of vectors used in mElo

You can get the results of different C of our MaxIn-mElo on an intransitive game by running:
```
sh run_c.sh
python C_plot.py
```

##### Comparison of different batch size $\tau$

You can get the results of different batch size $\tau$ of our MaxIn-Elo on an transitive game by running:

```
sh run_batch.sh
python batch_plot.py 0
```

Then the results of different batch size $\tau$ of our MaxIn-Elo on an intransitive game can be obtained by running:

```
python batch_plot.py 1
```

