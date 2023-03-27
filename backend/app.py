import json
import os
import numpy as np
import numpy.linalg as LA
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from unicodedata import normalize

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = os.environ.get("SQL_PASS")
MYSQL_PORT = 3306
MYSQL_DATABASE = "drinkdb"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

likes = []
dislikes = []
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

    # query_sql = f"""SELECT * FROM drinks WHERE drink_id IN (SELECT drink_id FROM ingredients WHERE LOWER( ingredient ) LIKE '%%{query.lower()}%%') limit 10"""
    query_sql = f"""SELECT DISTINCT ingredient FROM drinkdb.ingredients WHERE LOWER( ingredient ) LIKE '%%{query.lower()}%%' limit 10"""
    keys = ["ingredient"]
    data = mysql_engine.query_selector(query_sql)
    # keys = ["drink_id", "drink", "ingredients", "method"]
    return json.dumps([dict(zip(keys, i)) for i in data])
    # return data


def normalize_ingredient(ingredient):
    return normalize('NFC', ingredient.lower())


def ingredient_name_index():
    query_sql = f"""SELECT DISTINCT ingredient FROM drinkdb.ingredients"""
    data = mysql_engine.query_selector(query_sql)
    return {normalize_ingredient(ingredient[0]): i for i, ingredient in enumerate(data)}


def drink_ingredient_matrix(ingredient_index):
    query_sql = f"""SELECT * FROM drinkdb.ingredients"""
    data = [row for row in mysql_engine.query_selector(query_sql)]

    n_drinks = len(dict(data))
    n_ingredients = len(ingredient_index)

    result = np.zeros((n_drinks, n_ingredients))
    for drink_id, ingredient in data:
        result[drink_id,
               ingredient_index[normalize_ingredient(ingredient)]] = 1
    return result


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return sql_search(text)


def lookup_drink_by_id(id):
    query_sql = f"""SELECT * FROM drinkdb.drinks WHERE drink_id = {id} LIMIT 1"""
    data = mysql_engine.query_selector(query_sql)
    for row in data:
        return row


def vectorize_query(query, ingredient_index):
    result = np.zeros(len(ingredient_index))
    for ingredient in query:
        result[ingredient_index[ingredient]] = 1
    return result


def cosine_sim_ranking_ids(query_vec, d_i_matrix):
    q_norm = LA.norm(query_vec)
    doc_norms = LA.norm(d_i_matrix, axis=1)
    sims = np.matmul(d_i_matrix, query_vec)
    sims /= doc_norms
    sims /= q_norm

    return np.argsort(sims)[::-1]


def rocchio_update(likes_vec, dislikes_vec, d_i_matrix, alpha=1.0, beta=0.8, gamma=0.1, trim=False):
    relevant_drinks = d_i_matrix[np.where(
        np.any(np.logical_and(likes_vec, d_i_matrix), axis=1))]
    irrelevant_drinks = d_i_matrix[np.where(
        np.any(np.logical_and(dislikes_vec, d_i_matrix), axis=1))]
    rel = LA.norm(relevant_drinks, axis=0)
    nrel = LA.norm(irrelevant_drinks, axis=0)
    result = alpha * likes_vec + beta * rel + gamma * nrel
    if trim:
        result = np.maximum(result, 0.0)
    return result


@app.route("/likes", methods=["POST"])
def add_like():
    ingredient_index = ingredient_name_index()
    d_i_matrix = drink_ingredient_matrix(ingredient_index)

    likes = [normalize_ingredient(i)
             for i in request.args.get("likes").split(',')]
    if not likes:
        return json.dumps([])

    query_vec = vectorize_query(likes, ingredient_index)
    # query_vec = rocchio_update(query_vec, np.zeros(query_vec.shape), d_i_matrix,gamma=0)
    top_10 = cosine_sim_ranking_ids(query_vec, d_i_matrix)[:10]

    keys = ["drink_id", "drink", "ingredients", "method"]
    result = json.dumps([dict(zip(keys, lookup_drink_by_id(i)))
                        for i in top_10])
    print(likes)
    print(result)
    return result


@app.route("/dislikes", methods=["POST"])
def add_dislike():
    dislikes = request.args.get("dislikes")
    return json.dumps(dislikes)

# app.run(debug=True)
