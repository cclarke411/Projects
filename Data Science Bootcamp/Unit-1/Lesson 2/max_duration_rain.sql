WITH rainy as 
(
SELECT 
DATE(date) weather_date
From weather
WHERE events = 'Rain'
GROUP BY 1
),
rain_trips as (
SELECT
trip_id,
duration,
DATE(trips.start_date) rain_date
FROM trips
JOIN rainy
ON rainy.weather_date = DATE(trips.start_date)
ORDER BY duration DESC
)
SELECT 
rain_date,
MAX(duration) max_duration
FROM rain_trips
GROUP BY rain_date
ORDER BY max_duration DESC