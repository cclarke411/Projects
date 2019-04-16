SELECT MIN(calender_date) 
FROM sfo_calendar;

SELECT COUNT(*) 
FROM sfo_calendar; 

WITH days AS
 (
  SELECT listing_id, 
  COUNT(*) 
  FROM sfo_calendar 
  GROUP BY listing_id
 )
  SELECT * FROM days WHERE count <> 365;
  SELECT MIN(calender_date), MAX(calender_date) FROM sfo_calendar;
  
DROP TABLE calendar;

CREATE TABLE calendar AS
    SELECT *
    FROM sfo_calendar;

UPDATE calendar
SET price = REPLACE(price, '$', '')
WHERE available = 't';
UPDATE calendar
SET price = REPLACE(price, ',', '')
WHERE available = 't';
ALTER TABLE calendar ALTER COLUMN price TYPE FLOAT USING price::FLOAT;

WITH date_categories 
 AS(SELECT *, CASE WHEN calender_date <= DATE('2018-10-08') 
	 THEN 'SEP-OCT'WHEN calender_date <= DATE('2018-11-08') 
	THEN 'NOV-DEC'WHEN calender_date <= DATE('2019-01-08') 
	THEN 'DEC-JAN'WHEN calender_date <= DATE('2019-02-08') 
	THEN 'JAN-FEB'WHEN calender_date <= DATE('2019-03-08') 
	THEN 'FEB-MAR'WHEN calender_date <= DATE('2019-04-08') 
	THEN 'MAR-APR'WHEN calender_date <= DATE('2019-05-08') 
	THEN 'APR-MAY'WHEN calender_date <= DATE('2019-06-08') 
	THEN 'MAY-JUNE'WHEN calender_date <= DATE('2019-07-08') 
	THEN 'JUNE-JULY'WHEN calender_date <= DATE('2019-08-08') 
	THEN 'JULY-AUG'WHEN calender_date <= DATE('2019-09-08') 
	THEN 'AUG-SEP'ELSE 'UNKNOWN'END 
	AS duration 
	FROM calendar
   )
SELECT duration, 
AVG(price) average, 
COUNT(*) free_listings 
FROM date_categories WHERE available = 't'
GROUP BY duration;
