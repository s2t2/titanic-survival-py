
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

# Exploratory Queries

SQLite:

```sql
SELECT
  p.PassengerId as id
  ,upper(p.Name) as full_name
  ,case when instr(p.Name, "Mr.") then "MR"
        when instr(p.Name, "Mrs.") then "MRS"
        when instr(p.Name, "Miss.") then "MISS"
        when instr(p.Name, "Master.") then "MASTER"
        else "N/A"
        end salutation
  ,case when instr(p.Name, "Mr.") then 1
        when instr(p.Name, "Mrs.") then 1
        when instr(p.Name, "Miss.") then 0
        when instr(p.Name, "Master.") then 0
        else "N/A"
        end married

  ,upper(p.sex) as gender
  ,p.SibSp as sib_spouse_count
  ,p.Parch as parent_child_count
  ,case when p.Pclass = 1 then "UPPER"
        when p.Pclass = 2 then "MIDDLE"
        when p.Pclass = 3 then "LOWER"
        end ticket_class
  ,p.Fare as ticket_fare
  ,p.Ticket as ticket
  ,p.Cabin as cabin
  ,case when p.Embarked = "C" then upper("Cherbourg")
        when p.Embarked = "Q" then upper("Queenstown")
        when p.Embarked = "S" then upper("Southampton")
        else "N/A"
  end embarked_from

  ,p.survived

FROM passengers_train p
-- where ticket = "CA. 2343"
order by full_name

```
