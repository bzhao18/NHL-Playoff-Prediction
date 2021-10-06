## Group 2 Project Proposal - NHL Playoff Predictor
Bryan Zhao, Karishma Sabnani, Timothy Bang, Gabrielle Germanson, Noah Wallace

### Introduction/Background

The National Hockey League was formed in 1917 and spans 32 teams across the United States and Canada. In an average season, each team will play a total of 82 regular season games which generates a plethora of performance metrics on teams and players [1]. An important indicator of a successful hockey team is their ability to participate in the playoffs at the end of each season in April. 

### Problem Definition
In the NHL, along with many other sports, the winner of a game boils down to whoever can score the most points. There are many factors that contribute to a team’s point production and goals allowed, ranging from total shots accumulated and face-offs won, to penalties and total saves [2]. The dataset chosen contains such factors mentioned previously like goals, saves, penalties, and plays. The goal is to predict the rankings of the 16 teams that make it to the playoffs by the halfway point of the season.  


### Methods
Our current plan is to use three different supervised learning algorithms to predict NHL playoff standings. It will be important to identify models that will be most effective and feasible given the short timeline of the project, and some promising candidates include neural networks, decision trees, and logistic regression [3]. The team plans to assign members to develop the three models in parallel so that everyone can gain exposure through all steps of the process. After creating each model, the success rates of each model will be compared to determine the most effective prediction method.

We will train our model on predicting a season’s playoff standings based on each team’s performance metrics during the first half of their season. After training, the model will be able to predict the current or future seasons’ playoff teams by the halfway point of the season. Note, the 2020-2021 seasons are excluded, since they are in an adjusted season format for COVID.

### Timeline
<iframe style="width:100%; height:500px;overflow:auto;" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vR9UFpO27F-VLudRNhWWLRzRkvFFDGvQvN3FgBp6pab10n9RMswdYSTT3dT6Q8b-bMxAzKha9lLmio4/pubhtml?gid=0&amp;single=true&amp;widget=true&amp;headers=false"></iframe>
![Image of Timeline](/images/timeline.png)

### Potential Results and Discussion
A potential outcome of our project is that we have an algorithm that can make plausible predictions about the NHL playoffs. However, making predictions is difficult, even for experts. In a study published by ICT Innovations, artificial neural networks using different methods were tested to predict the outcome of NBA games. The best performing algorithm had an accuracy of 72.8% [4]. While those were predictions made on single games, we will be attempting to predict the outcome of all games in the second-half of the season. Consequently, making predictions on a larger scale bears the risk of having less precise predictions.

Moreover, considering we will work with data from the first half of the same season we are predicting playoff standings for, there are other challenges that threaten accuracy. These are unpredictable second-half incidents like player injuries and game cancellations.

### References
[1]    J. Weissbock, H. Viktor, and D. Inkpen, “Use of Performance Metrics to Forecast Success in the National Hockey League,” p. 10.\\
[2]    J. J. Heyne, A. J. Fenn, and S. Brook, “NHL Team Production,” Social Science Research Network, Rochester, NY, SSRN Scholarly Paper ID 942856, Oct. 2006. doi: 10.2139/ssrn.942856.\\
[3]    R. Bunker and T. Susnjak, “The Application of Machine Learning Techniques for Predicting Results in Team Sport: A Review,” ArXiv191211762 Cs Stat, Dec. 2019, Accessed: Oct. 03, 2021. [Online]. Available: http://arxiv.org/abs/1912.11762 \\
[4]    E. Zdravevski and A. Kulakov, “System for Prediction of the Winner in a Sports Game,” ICT Innov. 2009, pp. 55–63.
