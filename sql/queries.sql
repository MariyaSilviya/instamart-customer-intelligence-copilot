-- =========================================================================
-- Part A: Analytical SQL Queries for Instamart Copilot
-- =========================================================================

-- 1. Total Completed Orders and Gross Revenue by City
SELECT 
    c.city,
    COUNT(o.order_id) AS total_completed_orders,
    ROUND(SUM(o.order_amount), 2) AS total_revenue_inr,
    ROUND(AVG(o.order_amount), 2) AS avg_order_value_inr
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.city
ORDER BY total_revenue_inr DESC;


-- 2. Average Delivery Time (Minutes) per City for Completed Orders
SELECT 
    c.city,
    ROUND(AVG(o.delivery_minutes), 1) AS avg_delivery_minutes,
    COUNT(o.order_id) AS delivered_orders_count
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed' AND o.delivery_minutes > 0
GROUP BY c.city
ORDER BY avg_delivery_minutes ASC;


-- 3. Top 5 Best-Selling Product Categories by Total Quantity Sold
SELECT 
    p.category,
    SUM(o.quantity) AS total_units_sold,
    ROUND(SUM(o.order_amount), 2) AS category_revenue_inr
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY total_units_sold DESC;


-- 4. Monthly Order Volume and Revenue Trends
SELECT 
    strftime('%Y-%m', o.order_ts) AS order_month,
    COUNT(o.order_id) AS order_count,
    ROUND(SUM(o.order_amount), 2) AS monthly_revenue_inr
FROM orders o
WHERE o.status = 'completed'
GROUP BY order_month
ORDER BY order_month ASC;


-- 5. Customer Segmentation Summary (Order Frequency & Lifetime Value)
SELECT 
    c.customer_id,
    c.city,
    COUNT(o.order_id) AS lifetime_orders,
    ROUND(SUM(o.order_amount), 2) AS lifetime_spend
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
GROUP BY c.customer_id, c.city
ORDER BY lifetime_spend DESC
LIMIT 10;

-- 6. Order Status Distribution
SELECT
    status,
    COUNT(*) AS total_orders,
    ROUND(
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders),
        2
    ) AS percentage
FROM orders
GROUP BY status
ORDER BY total_orders DESC;


-- 7. Customer Revenue by City and Gender
SELECT
    c.city,
    c.gender,
    COUNT(DISTINCT c.customer_id) AS total_customers,
    COUNT(o.order_id) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS total_revenue_inr
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
GROUP BY c.city, c.gender
ORDER BY c.city, total_revenue_inr DESC;