from dash import Dash, dcc, html, dash_table
import sqlite3
import pandas as pd

DB_PATH=r"D:/Send to Katie/All Social Media/social_media.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM post;", conn)
conn.close()

app = Dash(__name__)

app.layout = html.Div([
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=20
    )
])

if __name__ == "__main__":
    app.run(debug=True)
    print("http://127.0.0.1:8050/")
