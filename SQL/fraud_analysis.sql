/*
Credit Card Fraud Analytics
---------------------------
Business questions:
1. What is the overall fraud rate?
2. How does fraud vary by transaction amount?
3. How does fraud vary by transaction hour?
4. Which hour + amount segments show elevated fraud rates?
5. What transaction value is associated with fraud?
6. How do fraud and legitimate transaction amounts differ?
*/




-- 1. Overall transaction KPIs

SELECT
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) AS legitimate_transactions,
    ROUND(
        100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*),
        4
    ) AS fraud_rate
FROM transactions;


-- 2. Fraud rate by transaction amount

SELECT
    CASE
        WHEN Amount < 50 THEN '0-50'
        WHEN Amount < 100 THEN '50-100'
        WHEN Amount < 250 THEN '100-250'
        WHEN Amount < 500 THEN '250-500'
        ELSE '500+'
    END AS amount_bucket,

    COUNT(*) AS total_transactions,

    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,

    ROUND(
        100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*),
        3
    ) AS fraud_rate

FROM transactions

GROUP BY
    CASE
        WHEN Amount < 50 THEN '0-50'
        WHEN Amount < 100 THEN '50-100'
        WHEN Amount < 250 THEN '100-250'
        WHEN Amount < 500 THEN '250-500'
        ELSE '500+'
    END;



    -- 3. Fraud rate by hour

SELECT
    MOD(FLOOR(Time / 3600), 24) AS hour,

    COUNT(*) AS total_transactions,

    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,

    ROUND(
        100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*),
        3
    ) AS fraud_rate

FROM transactions

GROUP BY MOD(FLOOR(Time / 3600), 24)

ORDER BY hour;



-- 4. Highest-risk hour + amount segments

SELECT
    MOD(FLOOR(Time / 3600), 24) AS hour,

    CASE
        WHEN Amount < 50 THEN '0-50'
        WHEN Amount < 100 THEN '50-100'
        WHEN Amount < 250 THEN '100-250'
        WHEN Amount < 500 THEN '250-500'
        ELSE '500+'
    END AS amount_bucket,

    COUNT(*) AS total_transactions,

    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,

    ROUND(
        100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*),
        3
    ) AS fraud_rate

FROM transactions

GROUP BY
    MOD(FLOOR(Time / 3600), 24),
    CASE
        WHEN Amount < 50 THEN '0-50'
        WHEN Amount < 100 THEN '50-100'
        WHEN Amount < 250 THEN '100-250'
        WHEN Amount < 500 THEN '250-500'
        ELSE '500+'
    END

HAVING COUNT(*) >= 100

ORDER BY fraud_rate DESC;




-- 5. Transaction value associated with fraud

SELECT
    SUM(Amount) AS total_transaction_value,

    SUM(
        CASE
            WHEN Class = 1 THEN Amount
            ELSE 0
        END
    ) AS fraudulent_transaction_value

FROM transactions;



-- 6. Average transaction amount by transaction type

SELECT
    CASE
        WHEN Class = 1 THEN 'Fraud'
        ELSE 'Legitimate'
    END AS transaction_type,

    COUNT(*) AS transaction_count,

    ROUND(AVG(Amount), 2) AS average_amount

FROM transactions

GROUP BY Class;




-- 7. Fraud transaction amount statistics

SELECT
    MIN(Amount) AS minimum_fraud_amount,
    MAX(Amount) AS maximum_fraud_amount,
    ROUND(AVG(Amount), 2) AS average_fraud_amount,
    SUM(Amount) AS total_fraud_amount

FROM transactions

WHERE Class = 1;




-- 8. High-value transactions and fraud rate

SELECT
    COUNT(*) AS high_value_transactions,

    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,

    ROUND(
        100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*),
        3
    ) AS fraud_rate

FROM transactions

WHERE Amount >= 500;