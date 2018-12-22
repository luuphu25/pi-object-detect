from flask import Flask
from flaskext.mysql import MySQL
from flask import render_template


app = Flask(__name__)
mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'test'
app.config['MYSQL_DATABASE_DB'] = 'testdb'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['MYSQL_DATABASE_PORT'] = 3333
mysql.init_app(app)


@app.route('/')
def users():
    conn = mysql.connect()
    cur = conn.cursor()
    cur.execute("SELECT * from detect")
    rv = cur.fetchall()
    print rv[0][0]
    return render_template('index.html', data=rv)
    #return str(rv)

if __name__ == '__main__':
    app.run(debug=True)