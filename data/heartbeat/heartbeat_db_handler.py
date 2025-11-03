# db_handler.py
import pymysql
from log import get_logger
from heartbeat_config import db_account
from heartbeat_config import Vehicle, Agency


class DB_Handler:
    def __init__(self):
        self.logger = get_logger("DB_LOG")
        self.db_conn, self.curs = self.connect_DB()

    def __del__(self):
        try:
            self.curs.close()
            self.db_conn.close()
        except:
            pass

    def connect_DB(self):
        try:
            conn = pymysql.connect(
                host=db_account['host'],
                user=db_account['user'],
                passwd=db_account['passwd'],
                db=db_account['db'],
                charset=db_account['charset'],
                cursorclass=pymysql.cursors.DictCursor
            )
            curs = conn.cursor()
            self.logger.info("DB connected")
            return conn, curs
        except Exception as e:
            self.logger.error(f"DB connection failed: {e}")
            raise

    def insert_heartbeat_tb(self, message):
        try:
            sql = """
                INSERT INTO heartbeat_tb (vehicle_id, agency_id, sensor_id, heart_beat, time_stamp)
                VALUES (%s, %s, %s, %s, now(3))
            """
            data = (
                Vehicle,#message[config.Vehicle],
                Agency,#message['agencyId'],
                message['sensor_id'],
                message['heart_beat'],           
            )
            self.curs.execute(sql, data)
            self.db_conn.commit()
            self.logger.info(f"Heartbeat Inserted: {data}")
        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Insert failed: {e}")
