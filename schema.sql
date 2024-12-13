CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    profile_picture TEXT,
    description TEXT
);
CREATE TABLE preferences (
    user_id TEXT,
    preferred TEXT,
    not_preferred TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
CREATE TABLE uncertain_pairs (
    user_id TEXT,
    img1 TEXT,
    img2 TEXT,
    UNIQUE(user_id, img1, img2),
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
