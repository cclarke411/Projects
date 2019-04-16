SELECT COUNT(DISTINCT sfo_listings.id) AS list_cnt,
       sfo_listings.neighbourhood AS list_neigh
FROM sfo_listings
GROUP BY list_neigh
ORDER BY list_cnt DESC
