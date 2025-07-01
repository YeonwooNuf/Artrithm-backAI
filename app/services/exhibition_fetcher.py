from app.db.database import get_connection
import pymysql

def fetch_all_exhibitions():
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)  # ✅ DictCursor 사용

    cursor.execute("""
        SELECT e.id, e.title, e.description, GROUP_CONCAT(k.name SEPARATOR ', ') AS keywords
        FROM exhibitions e
        LEFT JOIN exhibition_keywords ek ON e.id = ek.exhibition_id
        LEFT JOIN keywords k ON ek.keyword_id = k.id
        GROUP BY e.id
    """)

    rows = cursor.fetchall()
    exhibitions = []
    for row in rows:
        exhibitions.append({
            "id": row["id"],
            "title": row["title"],
            "description": row["description"] or "",
            "keywords": row["keywords"] or ""
        })

    return exhibitions
