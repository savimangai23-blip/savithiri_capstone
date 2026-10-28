-- Q1: WHERE then HAVING
SELECT customer_id
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: three-table join
SELECT c.city, p.category, SUM(o.order_amount) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: scalar subquery
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
GROUP BY c.customer_id, c.city
HAVING COUNT(o.order_id) = 0
ORDER BY c.customer_id;

-- Q5: CTE above average spend
WITH total_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_completed_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
), avg_total AS (
    SELECT AVG(total_completed_spend) AS avg_spend
    FROM total_spend
)
SELECT ts.customer_id, ts.total_completed_spend
FROM total_spend ts
CROSS JOIN avg_total a
WHERE ts.total_completed_spend > a.avg_spend
ORDER BY ts.total_completed_spend DESC;

-- Q6: ranking window functions
WITH customer_spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_spend_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS city_row_number
FROM customer_spend
ORDER BY city, city_row_number;

-- Q7: running daily revenue
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, revenue,
       SUM(revenue) OVER (ORDER BY day) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
