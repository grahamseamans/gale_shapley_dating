import sqlite3

DB_FILE = "preferences.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            profile_picture TEXT,
            description TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS preferences (
            user_id TEXT,
            preferred TEXT,
            not_preferred TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS uncertain_pairs (
            user_id TEXT,
            img1 TEXT,
            img2 TEXT,
            UNIQUE(user_id, img1, img2),
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        """
    )
    conn.commit()
    return conn


def register_user(conn, user_id, description, profile_picture):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    if cursor.fetchone():
        return False
    cursor.execute(
        "INSERT INTO users (user_id, profile_picture, description) VALUES (?, ?, ?)",
        (user_id, profile_picture, description),
    )
    conn.commit()
    return True


def record_preference(conn, user_id, preferred, not_preferred):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO preferences (user_id, preferred, not_preferred) VALUES (?, ?, ?)",
        (user_id, preferred, not_preferred),
    )
    conn.commit()


def remove_uncertain_pair(conn, user_id, img1, img2):
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM uncertain_pairs WHERE user_id = ? AND img1 = ? AND img2 = ?",
        (user_id, img1, img2),
    )
    conn.commit()


def get_all_users(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM users")
    return [row[0] for row in cursor.fetchall()]


def get_user_profile(conn, user_id):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT profile_picture, description FROM users WHERE user_id = ?", (user_id,)
    )
    return cursor.fetchone()


def update_user_description(conn, user_id, new_description):
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET description = ? WHERE user_id = ?",
        (new_description, user_id),
    )
    conn.commit()


def get_uncertain_pairs(conn, user_id):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT img1, img2 FROM uncertain_pairs WHERE user_id = ?", (user_id,)
    )
    return cursor.fetchall()


def store_uncertain_pairs(conn, user_id, uncertain_pairs):
    cursor = conn.cursor()
    for _, img1_path, img2_path in uncertain_pairs:
        cursor.execute(
            "INSERT OR IGNORE INTO uncertain_pairs (user_id, img1, img2) VALUES (?, ?, ?)",
            (user_id, img1_path, img2_path),
        )
    conn.commit()


def get_preferences(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM preferences")
    return cursor.fetchall()


def get_label_count(conn, user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM preferences WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    return result[0] if result else 0
