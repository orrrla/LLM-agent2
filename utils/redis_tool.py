# -*- coding: utf-8 -*-
import redis

from config.runtime import get_redis_settings


class RedisDBConfig:
    settings = get_redis_settings()
    HOST = settings.host
    PORT = settings.port
    DBID = settings.db
 
def operator_status(func):
    '''get operatoration status
    '''
    def gen_status(*args, **kwargs):
        error, result = None, None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error = str(e)
 
        return result 
 
    return gen_status
 
class RedisClient(object):
    def __init__(self):
        if not hasattr(RedisClient, 'pool'):
            RedisClient.create_pool()
        self._connection = redis.Redis(connection_pool = RedisClient.pool)
 
    @staticmethod
    def create_pool():
        RedisClient.pool = redis.ConnectionPool(
                host = RedisDBConfig.HOST,
                port = RedisDBConfig.PORT,
                db   = RedisDBConfig.DBID,
                decode_responses=True)
 
    @operator_status
    def set(self, key, value, ex=None):
        '''set data with (key, value)
        '''
        return self._connection.set(key, value, ex=ex)
 
    @operator_status
    def get(self, key):
        '''get data by key
        '''
        return self._connection.get(key)
 
 
 
if __name__ == '__main__':
    print(RedisClient().set('Testkey', "Simple Test", ex=10))
    print(RedisClient().get('Testkey'))
