'''1757'''
# Write your MySQL query statement below
SELECT product_id FROM Products
WHERE low_fats = 'Y' AND recyclable = 'Y';

'''584'''
# Write your MySQL query statement below
SELECT name FROM Customer
WHERE referee_id != 2 OR referee_id IS NULL;

'''595'''
# Write your MySQL query statement below
SELECT name, population, area FROM World
WHERE area >= 3000000 OR population >= 25000000;

'''1148'''
# Write your MySQL query statement below
SELECT DISTINCT viewer_id AS id FROM Views
WHERE viewer_id = author_id
ORDER BY id

'''1683'''
# Write your MySQL query statement below
SELECT tweet_id FROM Tweets
WHERE LENGTH(content) > 15;

'''1378'''
SELECT unique_id, name
FROM Employees
LEFT JOIN EmployeeUNI ON Employees.id = EmployeeUNI.id

'''1068'''
SELECT product_name, year, price
FROM Sales
JOIN Product ON Sales.product_id = Product.product_id

'''1581'''
SELECT customer_id, Count(Visits.visit_id) as count_no_trans
FROM Visits
LEFT JOIN Transactions ON Visits.visit_id = Transactions.visit_id
WHERE Transactions.visit_id IS NULL
GROUP BY customer_id

'''197'''
SELECT w2.id
FROM Weather w1
JOIN Weather w2 ON datediff(w2.recordDate, w1.recordDate) = 1
WHERE w2.Temperature > w1.Temperature