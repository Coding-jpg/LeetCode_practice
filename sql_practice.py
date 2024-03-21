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