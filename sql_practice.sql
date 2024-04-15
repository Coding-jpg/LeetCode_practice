---1757---
# Write your MySQL query statement below
SELECT product_id FROM Products
WHERE low_fats = 'Y' AND recyclable = 'Y';

---584---
# Write your MySQL query statement below
SELECT name FROM Customer
WHERE referee_id != 2 OR referee_id IS NULL;

---595---
# Write your MySQL query statement below
SELECT name, population, area FROM World
WHERE area >= 3000000 OR population >= 25000000;

---1148---
# Write your MySQL query statement below
SELECT DISTINCT viewer_id AS id FROM Views
WHERE viewer_id = author_id
ORDER BY id

---1683---
# Write your MySQL query statement below
SELECT tweet_id FROM Tweets
WHERE LENGTH(content) > 15;

---1378---
SELECT unique_id, name
FROM Employees
LEFT JOIN EmployeeUNI ON Employees.id = EmployeeUNI.id

---1068---
SELECT product_name, year, price
FROM Sales
JOIN Product ON Sales.product_id = Product.product_id

---1581---
SELECT customer_id, Count(Visits.visit_id) as count_no_trans
FROM Visits
LEFT JOIN Transactions ON Visits.visit_id = Transactions.visit_id
WHERE Transactions.visit_id IS NULL
GROUP BY customer_id

---197---
SELECT w2.id
FROM Weather w1
JOIN Weather w2 ON datediff(w2.recordDate, w1.recordDate) = 1
WHERE w2.Temperature > w1.Temperature

---1661---
# Write your MySQL query statement below
select
a1.machine_id, round(avg(a2.timestamp - a1.timestamp), 3) as processing_time
from Activity as a1 join Activity as a2 on
a1.machine_id=a2.machine_id and
a1.process_id=a2.process_id and
a1.activity_type='start' and
a2.activity_type='end'
group by machine_id

---577---
# Write your MySQL query statement below
select name, bonus
from Employee left join Bonus on Employee.empId = Bonus.empId
where bonus < 1000 or bonus is null

---1280---
# Write your MySQL query statement below
select s.student_id, s.student_name, sub.subject_name, IFNULL(grouped.attended_exams, 0) as attended_exams
from Students s
cross join Subjects sub
left join(
    select student_id, subject_name, count(*) as attended_exams
    from Examinations
    group by student_id, subject_name
) grouped
on s.student_id = grouped.student_id and sub.subject_name = grouped.subject_name
order by s.student_id, sub.subject_name

---570---
select name
from Employee
where id in (
    select distinct ManagerId
    from Employee
    group by ManagerId
    having count(ManagerId) >= 5
)

---1943---
# Write your MySQL query statement below
select s.user_id, round(ifnull(avg(c.action='confirmed'), 0), 2) as confirmation_rate
from Confirmations as c
right join Signups as s on c.user_id = s.user_id 
group by s.user_id

---620---
select  id, movie, description, rating
from cinema
where description <> 'boring'
    and (id % 2) <> 0
order by rating desc

---1251---
select Prices.product_id, ifnull(round(sum(price*units)/sum(units), 2), 0) as average_price
from Prices
left join UnitsSold
on Prices.product_id = UnitsSold.product_id and
    UnitsSold.purchase_date between Prices.start_date and Prices.end_date
group by product_id

---1075---
select project_id, round(avg(experience_years), 2) as average_years
from Project
join Employee
on Project.employee_id = Employee.employee_id
group by project_id

-- 1633
select contest_id, round(count(user_id)/(select count(user_id) from Users), 4) * 100 as percentage
from Register
group by contest_id
order by percentage desc, contest_id asc

-- 1211
WITH poor AS (
    SELECT 
        query_name,
        SUM(CASE WHEN rating < 3 THEN 1 ELSE 0 END) AS poor_ratings,
        COUNT(*) AS total
    FROM Queries
    GROUP BY query_name
)

SELECT 
    p.query_name, 
    ROUND(AVG(q.rating / q.position), 2) AS quality, 
    ROUND((poor_ratings * 100.0) / total, 2) AS poor_query_percentage
FROM poor p
JOIN Queries q ON p.query_name = q.query_name
GROUP BY p.query_name;

-- 1193
# Write your MySQL query statement below
select date_format(trans_date, '%Y-%m') as month,
    country,
    count(*) as trans_count,
    count(if(state='approved', 1, NULL)) as approved_count,
    sum(amount) as trans_total_amount,
    sum(if(state='approved', amount, 0)) as approved_total_amount
from Transactions
group by month, country

-- 1174
select round (
    sum(order_date = customer_pref_delivery_date) * 100 /
    count(*),
    2
) as immediate_percentage
from Delivery
where (customer_id, order_date) in (
    select customer_id, min(order_date)
    from delivery
    group by customer_id
)

-- 550
select IFNULL(round(count(distinct(Result.player_id)) / count(distinct(Activity.player_id)), 2), 0) as fraction
from (
  select Activity.player_id as player_id
  from (
    select player_id, DATE_ADD(MIN(event_date), INTERVAL 1 DAY) as second_date
    from Activity
    group by player_id
  ) as Expected, Activity
  where Activity.event_date = Expected.second_date and Activity.player_id = Expected.player_id
) as Result, Activity

-- 2356
select teacher_id, count(distinct subject_id) as cnt
from Teacher
group by teacher_id

-- 1141
select activity_date as day, count(distinct user_id) as active_users
from Activity
where datediff('2019-07-27', activity_date) < 30 and datediff('2019-07-27', activity_date) >= 0
group by activity_date

-- 1084
select distinct s.product_id, p.product_name
from Sales s
join Product p
on p.product_id = s.product_id
where s.sale_date >= '2019-01-01' and s.sale_date <= '2019-03-31' and
    s.product_id not in (
        select distinct s1.product_id
        from Sales s1
        where s1.sale_date < '2019-01-01' or s1.sale_date > '2019-03-31'
    )

-- 596
select class
from Courses
group by class
having count(student) >= 5

-- 1729
select user_id, count(distinct follower_id) as followers_count
from Followers
group by user_id

--619
select max(num) as num
from (
    select num
    from MyNumbers
    group by num
    having count(num) = 1
) as Nums

-- 1045
select customer_id
from Customer
group by customer_id
having count(distinct product_key) = (
    select count(product_key) from Product
)

-- 1731
SELECT 
  E.reports_to AS employee_id, 
  (SELECT name FROM Employees WHERE employee_id = E.reports_to) AS name,
  COUNT(*) AS reports_count,
  ROUND(AVG(E.age)) AS average_age
FROM 
  Employees E
WHERE 
  E.reports_to IS NOT NULL
GROUP BY 
  E.reports_to
ORDER BY employee_id asc

-- 1789
select employee_id, department_id
from Employee
group by employee_id
having count(department_id) = 1
union
select employee_id, department_id
from Employee
where primary_flag = 'Y'

-- 610
SELECT x, y, z,
       CASE
           WHEN (x + y > z) AND (x + z > y) AND (y + z > x) THEN 'Yes'
           ELSE 'No'
       END AS Triangle
FROM Triangle;

-- 180
SELECT DISTINCT
    l1.Num AS ConsecutiveNums
FROM
    Logs l1,
    Logs l2,
    Logs l3
WHERE
    l1.Id = l2.Id - 1
    AND l2.Id = l3.Id - 1
    AND l1.Num = l2.Num
    AND l2.Num = l3.Num

-- 1164
SELECT DISTINCT product_id, IF(filter_date IS NULL, 10, new_price) AS price
FROM (
  SELECT *, RANK() OVER(PARTITION BY product_id ORDER BY filter_date DESC) AS RANKING
  FROM (
    SELECT *, IF(change_date > '2019-08-16', NULL, change_date) AS filter_date
    FROM Products
  ) T
) TT
WHERE TT.RANKING = 1

-- 1204
select person_name
from (
    select person_name, sum(weight) over(order by turn asc) as sum_weight
    from Queue
) as sum_table
where sum_weight <= 1000
order by sum_weight desc
limit 1

-- 1907
SELECT 
    'Low Salary' AS category,
    SUM(CASE WHEN income < 20000 THEN 1 ELSE 0 END) AS accounts_count
FROM 
    Accounts
    
UNION
SELECT  
    'Average Salary' category,
    SUM(CASE WHEN income >= 20000 AND income <= 50000 THEN 1 ELSE 0 END) 
    AS accounts_count
FROM 
    Accounts

UNION
SELECT 
    'High Salary' category,
    SUM(CASE WHEN income > 50000 THEN 1 ELSE 0 END) AS accounts_count
FROM 
    Accounts

-- 1978
select a.employee_id
from Employees a left join Employees b
on a.manager_id = b.employee_id
where a.manager_id is not null and b.employee_id is null and a.salary < 30000
order by a.employee_id

-- 626
select 
    (case
        when mod(id, 2) != 0 and counts != id then id + 1
        when mod(id, 2) != 0 and counts = id then id
        else id-1
    end) as id,
    student
from
    seat,
    (select
        count(*) as counts
    from
        seat) as seat_counts
order by id asc

-- 1341
(select u.name as results
from MovieRating mr 
left join Users u using(user_id) 
group by u.user_id
order by count(movie_id) desc, u.name asc
limit 1)

union all

(select m.title results
from MovieRating mr 
left join Movies m using(movie_id)
where date_format(mr.created_at,'%Y-%m') = '2020-02'
group by m.movie_id
order by avg(rating) desc, m.title asc
limit 1)

-- 1321
select distinct LL.visited_on,
                LL.sum_amount as amount,
                round(LL.sum_amount/7, 2) as average_amount
from (
    select visited_on, 
        sum(TT.amount) over(
        order by TT.visited_on
        rows 6 preceding 
    ) as sum_amount
    from (
        select visited_on,
                sum(amount) as amount
        from Customer
        group by visited_on
    ) TT
) LL
where datediff(visited_on, (select min(visited_on) from Customer)) >= 6