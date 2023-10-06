import sqlite3

connect = sqlite3.connect('questions.db')
c = connect.cursor()
table_name = 'ict'
c.execute(f'''CREATE TABLE {table_name} (
            id INTEGER ,
            question TEXT,
            v1 TEXT,
            v2 TEXT,
            v3 TEXT,
            v4 TEXT)
            ''')
connect.commit()
connect.close()
