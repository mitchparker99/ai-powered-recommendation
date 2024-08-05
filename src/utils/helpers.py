import redis


def cache_recommendations(user_id, recommendations):
    cache = redis.StrictRedis(host='localhost', port=6379, db=0)
    cache.set(user_id, recommendations)
