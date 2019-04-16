SELECT MAX(price),
  neighbourhood ,
  host_id,
  host_name,
  room_type,
  minimum_nights,
  number_of_reviews
FROM sfo_listings
GROUP BY sfo_listings.neighbourhood,neighbourhood,host_id,host_name,room_type,minimum_nights,number_of_reviews,price
ORDER BY price DESC
LIMIT 1
