SELECT 
	DISTINCT t.duration,
	t.trip_id,
	DATE(t.start_date)
FROM trips t
LEFT JOIN 
 	weather
ON
	DATE(t.start_date) = DATE(weather.date)
WHERE
    weather.precipitationin > 0
ORDER BY duration DESC
LIMIT 3