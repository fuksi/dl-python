import pendulum
import psycopg2

class Db:
    def __init__(self):
        self.host = ''
        self.db = ''
        self.user = ''
        self.pw = ''

    def connect(self):
        return psycopg2.connect(host=self.host,database=self.db, user=self.user, password=self.pw)

    def get_simple_moving_avg(self, periods, current_date, symbols):
        results = []
        with self.connect() as con:
            with con.cursor() as cur: 
                for symbol in symbols:
                    for period in periods:
                        start = current_date.subtract(days=period)
                        cur.execute("""
                            SELECT avg(close) FROM candles 
                            WHERE time BETWEEN %s and %s
                            AND symbol = %s
                        """, (start.to_iso8601_string(), current_date.to_iso8601_string(), symbol))

                        result = cur.fetchall()
                        assert len(result) == 1

                        results.append(result[0][0])

        
        return results

    def get_today_tmr_close(self, current_date, symbols):
        today = []
        tmr = []
        end_of_today = current_date.end_of('day')
        end_of_tmr = current_date.add(days=1).end_of('day')
        with self.connect() as con:
            with con.cursor() as cur:
                for symbol in symbols:
                    cur.execute("""
                        SELECT close FROM candles WHERE
                        time > %s  
                        AND symbol = %s 
                        ORDER BY time asc
                        LIMIT 1
                    """, (end_of_today.to_iso8601_string(), symbol))

                    result = cur.fetchall()
                    assert len(result) == 1

                    today.append(result[0][0])

                    cur.execute("""
                        SELECT close FROM candles WHERE
                        time > %s  
                        AND symbol = %s
                        ORDER BY time asc
                        LIMIT 1
                    """, (end_of_tmr.to_iso8601_string(), symbol))

                    result = cur.fetchall()
                    assert len(result) == 1

                    tmr.append(result[0][0])

                return (today, tmr)


