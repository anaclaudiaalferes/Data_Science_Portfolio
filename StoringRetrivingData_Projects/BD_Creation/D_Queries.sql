use `srd_ims`;
#1.
#List all the customers
SELECT * FROM customer;
#List all the dates 
SELECT * FROM date;
#Products/services bought by these customers in a range of two dates.
SELECT p.PRODUCTS_NAME
FROM products as p
JOIN  `sales-products` sp on sp.PRODUCTS_ID = p.PRODUCTS_ID
JOIN sales s on sp.SALES_ID = s.SALES_ID
JOIN date d on s.DATE_ID = d.DATE_ID
WHERE d.DATE BETWEEN '2019-12-25' and '2019-12-27';

#2
#List the best three clients.
SELECT SUM(s.TOTAL_AMOUNT) as Total_Spend, c.CUSTOMER_NAME
FROM  sales as s
JOIN customer c on s.CUSTOMER_ID = c.CUSTOMER_ID
GROUP BY c.CUSTOMER_ID
ORDER BY Total_Spend DESC
LIMIT 3;

#3
# Average Money per Day
SELECT avg(`TOTAL_AMOUNT`) as `Average Money`, `DAY`, `MONTH`
FROM `sales` as s
JOIN `date` as d ON d.DATE_ID=s.DATE_ID
group by `DAY`;

#Average Money per Week
SELECT avg(`TOTAL_AMOUNT`) as `Average Money`, `WEEK`, `CALENDAR_YEAR`
FROM `sales`as s
JOIN `date` as d ON d.DATE_ID=s.DATE_ID
group by `WEEK`;

#Average Money per Month
SELECT avg(`TOTAL_AMOUNT`), `MONTH`
FROM `sales` as s
JOIN `date` as d ON d.DATE_ID=s.DATE_ID
group by `MONTH`;

#Average Money per Year
SELECT avg(`TOTAL_AMOUNT`), `CALENDAR_YEAR`
FROM `sales`as s
JOIN `date`as d ON d.DATE_ID=s.DATE_ID
group by `CALENDAR_YEAR`;

#4
#Get the total sales per location (country).
SELECT SUM(s.TOTAL_AMOUNT) as Total_Spend, a.COUNTRY
FROM  sales as s
JOIN address a on s.CUSTOMER_ID = a.CUSTOMER_ID
GROUP BY a.COUNTRY
ORDER BY Total_Spend DESC;

#Get the total sales per location (city).
SELECT SUM(s.TOTAL_AMOUNT) as Total_Spend, a.CITY
FROM  sales as s
JOIN address a on s.CUSTOMER_ID = a.CUSTOMER_ID
GROUP BY a.CITY
ORDER BY Total_Spend DESC;


#5
#List all the locations where products/services where sold and the product have customer’s ratings.
SELECT  a.CITY
FROM  sales as s
JOIN address a on s.CUSTOMER_ID = a.CUSTOMER_ID
GROUP BY a.CITY;

SELECT p.PRODUCTS_NAME,p.PRODUCT_DESCRIPTION, avg(r.RATING)
FROM `sales-products` as sp
JOIN sales as s ON sp.SALES_ID=s.SALES_ID
JOIN products as p ON SP.PRODUCTS_ID=p.PRODUCTS_ID
JOIN rating as r ON s.SALES_ID=r.SALES_ID
where r.RATING_ID in (select RATING_ID from (
    select
        RATING_ID,
        DATE,
        SALES_ID,
        row_number() over(partition by SALES_ID order by DATE desc) as row
    from
        rating
) test
where test.row = 1)
group by p.PRODUCTS_ID;

#E Create a view called INVOICE that contains all the fields in the invoice below.
CREATE VIEW INVOICE AS
SELECT s.SALES_ID, d.DATE, c.CUSTOMER_NAME, ad.STREET, ad.CITY, ad.COUNTRY, ad.ZIP_CODE,
p.PRODUCT_DESCRIPTION,p.UNIT_PRICE, sp.QUANTITY, sp.SUBTOTAL as SUBTOTAL_PRODUCT, s.SUBTOTAL, s.TAX, s.TOTAL_AMOUNT
FROM sales as s 
JOIN date as d on s.DATE_ID = d.DATE_ID
JOIN customer as c on s.CUSTOMER_ID = c.CUSTOMER_ID
JOIN address as ad on s.ADDRESS_ID = ad.ADDRESS_ID
JOIN `sales-products`as sp on s.SALES_ID=sp.SALES_ID
JOIN products as p on sp.PRODUCTS_ID=p.PRODUCTS_ID;

