
import sqlite3

conn = sqlite3.connect('Cache/proximity_vectors.sqlite3')
c = conn.cursor()
c.execute('SELECT * FROM results LIMIT 10')
print(c.fetchall())
c.close()
