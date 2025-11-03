import pymysql

# id 'root'@'localhost'     passwd #1 1234, #2 ? #3 keti1234
# id 'podo'@'%'             passwd podo1234
# id 'keti'@'%'             passwd keti1234
# db : dbdb
# host : #1 : 192.168.10.110:3306 (subnet : 255.255.255.0, Gateway : 192.168.10.254)

db_account = {
    "host": "192.168.10.110",
    "user": "keti",
    "passwd": "keti1234",
    "db": "dbdb",
    "charset": "utf8"
}


#db_account = {
#    "user": pymysql.connect(
#        user = 'root',
#        passwd =  "keti1234",
#        host = "127.0.0.1",   # 로컬 DB
#        db = "dbdb",
#        charset = "utf8"
#    )
#}


# Heartbeat client data configuration
Vehicle = "MV-04"
Agency = "KETI"

