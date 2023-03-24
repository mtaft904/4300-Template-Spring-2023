import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = "YOURPASSWORDHERE"
MYSQL_PORT = 3306
MYSQL_DATABASE = "drinkdb"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

likes = []
# Sample search, the LIKE operator in this case is hard-coded,
# but if you decide to use SQLAlchemy ORM framework,
# there's a much better and cleaner way to do this


def sql_search(query):
    # query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
    # keys = ["id", "title", "descr"]
    # data = mysql_engine.query_selector(query_sql)
    # return json.dumps([dict(zip(keys, i)) for i in data])

    # query_sql = f""""SELECT * FROM ingredients WHERE LOWER( ingredient ) LIKE '{query.lower()}' limit 10"""
    # matching_drink_ids = [id for id,
    #                       _ in mysql_engine.query_selector(query_sql)]
    # sql_descriptions = f""""SELECT * FROM drinks WHERE drink_id IN '%%{matching_drink_ids}%%'"""
    # keys = ["drink_id", "drink", "ingredients"]
    # data = mysql_engine.query_selector(sql_descriptions)
    # return json.dumps([dict(zip(keys, i)) for i in data])

    #query_sql = f"""SELECT * FROM drinks WHERE drink_id IN (SELECT drink_id FROM ingredients WHERE LOWER( ingredient ) LIKE '%%{query.lower()}%%') limit 10"""
    query_sql = f"""SELECT DISTINCT ingredient FROM ingredients WHERE LOWER( ingredient ) LIKE '%%{query.lower()}%%' limit 10"""
    keys = ["ingredient"]
    data = mysql_engine.query_selector(query_sql)
    #keys = ["drink_id", "drink", "ingredients", "method"]
    return json.dumps([dict(zip(keys, i)) for i in data])
    #return data


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return sql_search(text)

@app.route("/likes", methods=["POST"])
def add_like():
    likes = request.args.get("likes")
    return json.dumps(likes)

# app.run(debug=True)
