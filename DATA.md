
# Data Dictionary

Data Source: https://www.kaggle.com/c/titanic

  + Training set (891 rows)
  + Test set (419 rows)

Columns / Features:

Variable | 	Definition	| Key
--- | --- | ---
`survival`	| Survival	| 0 = No, 1 = Yes
`pclass`	| Ticket class, a proxy for socio-economic status | 1 = 1st (Upper), 2 = 2nd (Middle), 3 = 3rd (Lower)
`sex`	| Sex / Gender	|
`age`	| Age in years	| Fractional if less than 1. If estimated, is in the form of x.5
`sibsp`	| # of siblings / spouses aboard the Titanic	| Sibling = brother, sister, stepbrother, stepsister. Spouse = husband, wife
`parch`	| # of parents / children aboard the Titanic. 	| Parent = mother, father. Child = daughter, son, stepdaughter, stepson. Some children travelled only with a nanny, therefore parch=0 for them.
`ticket`	| Ticket number	|
`fare` |	Passenger fare	|
`cabin` |	Cabin number
`embarked`	| Port of Embarkation	| C = Cherbourg, Q = Queenstown, S = Southampton
