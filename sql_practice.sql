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