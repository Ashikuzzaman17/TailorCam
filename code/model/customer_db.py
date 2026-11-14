"""
TailorCam Customer Database
SQLite-backed storage for customers, measurements, orders, inventory, and staff.
"""
import os
import json
import sqlite3
import datetime


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'tailorcam.db')


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist (safe to call on every startup)."""
    with _connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            phone       TEXT DEFAULT '',
            email       TEXT DEFAULT '',
            gender      TEXT DEFAULT 'unspecified',
            age         INTEGER DEFAULT NULL,
            weight_kg   REAL    DEFAULT NULL,
            height_cm   REAL    DEFAULT NULL,
            notes       TEXT DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS measurements (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id     INTEGER NOT NULL,
            scan_date       TEXT NOT NULL,
            shoulder_width  REAL, chest         REAL,
            waist           REAL, hip           REAL,
            back_length     REAL, shirt_length  REAL,
            sleeve_length   REAL, neck          REAL,
            height_cm       REAL,
            garment_type    TEXT DEFAULT 't_shirt',
            notes           TEXT DEFAULT '',
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );

        CREATE TABLE IF NOT EXISTS orders (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id     INTEGER NOT NULL,
            measurement_id  INTEGER,
            garment_type    TEXT NOT NULL,
            style_notes     TEXT DEFAULT '',
            fabric          TEXT DEFAULT '',
            colour          TEXT DEFAULT '',
            price           REAL DEFAULT 0.0,
            advance_paid    REAL DEFAULT 0.0,
            order_date      TEXT NOT NULL,
            delivery_date   TEXT DEFAULT '',
            status          TEXT DEFAULT 'received',
            assigned_to     TEXT DEFAULT '',
            FOREIGN KEY (customer_id)    REFERENCES customers(id),
            FOREIGN KEY (measurement_id) REFERENCES measurements(id)
        );

        CREATE TABLE IF NOT EXISTS inventory (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            category    TEXT DEFAULT 'fabric',
            qty         REAL DEFAULT 0.0,
            unit        TEXT DEFAULT 'meters',
            min_qty     REAL DEFAULT 2.0,
            colour      TEXT DEFAULT '',
            price_per   REAL DEFAULT 0.0,
            notes       TEXT DEFAULT '',
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS staff_members (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            role        TEXT DEFAULT 'Tailor',
            active      INTEGER DEFAULT 1,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id  INTEGER NOT NULL,
            customer_id     INTEGER NOT NULL,
            feedback_date   TEXT NOT NULL,
            field           TEXT NOT NULL,
            measured_cm     REAL,
            actual_cm       REAL,
            error_cm        REAL GENERATED ALWAYS AS (actual_cm - measured_cm) VIRTUAL,
            tailor_note     TEXT DEFAULT '',
            FOREIGN KEY (measurement_id) REFERENCES measurements(id),
            FOREIGN KEY (customer_id)    REFERENCES customers(id)
        );
        """)

    # ── Backwards-compat: add assigned_to column if missing ───────────────────
    with _connect() as conn:
        try:
            conn.execute("ALTER TABLE orders ADD COLUMN assigned_to TEXT DEFAULT ''")
        except Exception:
            pass  # column already exists

    # ── Backwards-compat: add photo_path column to customers if missing ────────
    with _connect() as conn:
        try:
            conn.execute("ALTER TABLE customers ADD COLUMN photo_path TEXT DEFAULT ''")
        except Exception:
            pass  # column already exists

    # ── Backwards-compat: ensure feedback table exists (added in v2) ──────────
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                measurement_id  INTEGER NOT NULL,
                customer_id     INTEGER NOT NULL,
                feedback_date   TEXT NOT NULL,
                field           TEXT NOT NULL,
                measured_cm     REAL,
                actual_cm       REAL,
                error_cm        REAL GENERATED ALWAYS AS (actual_cm - measured_cm) VIRTUAL,
                tailor_note     TEXT DEFAULT '',
                FOREIGN KEY (measurement_id) REFERENCES measurements(id),
                FOREIGN KEY (customer_id)    REFERENCES customers(id)
            )
        """)

    # ── Seed default staff if empty ────────────────────────────────────────────
    with _connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM staff_members").fetchone()[0]
        if count == 0:
            now = _now()
            conn.execute(
                "INSERT INTO staff_members (name, role, active, created_at) VALUES (?,?,1,?)",
                ('Main Tailor', 'Senior Tailor', now)
            )

    print('[DB] Database initialised:', os.path.abspath(DB_PATH))


# ── Customer CRUD ──────────────────────────────────────────────────────────────

def add_customer(name, phone='', email='', gender='unspecified',
                 age=None, weight_kg=None, height_cm=None, notes=''):
    now = _now()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO customers (name,phone,email,gender,age,weight_kg,"
            "height_cm,notes,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (name, phone, email, gender, age, weight_kg, height_cm, notes, now, now)
        )
        return cur.lastrowid


def update_customer(cid, **kwargs):
    kwargs['updated_at'] = _now()
    cols = ', '.join(f'{k}=?' for k in kwargs)
    vals = list(kwargs.values()) + [cid]
    with _connect() as conn:
        conn.execute(f'UPDATE customers SET {cols} WHERE id=?', vals)


def get_customer(cid):
    with _connect() as conn:
        row = conn.execute('SELECT * FROM customers WHERE id=?', (cid,)).fetchone()
        return dict(row) if row else None


def search_customers(query='', limit=50):
    with _connect() as conn:
        if query:
            rows = conn.execute(
                "SELECT * FROM customers WHERE name LIKE ? OR phone LIKE ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (f'%{query}%', f'%{query}%', limit)
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT * FROM customers ORDER BY updated_at DESC LIMIT ?',
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def delete_customer(cid):
    with _connect() as conn:
        conn.execute('DELETE FROM measurements WHERE customer_id=?', (cid,))
        conn.execute('DELETE FROM orders WHERE customer_id=?', (cid,))
        conn.execute('DELETE FROM customers WHERE id=?', (cid,))


def save_customer_photo(customer_id, photo_path: str):
    """Store the file path of a customer's profile photo."""
    with _connect() as conn:
        conn.execute(
            "UPDATE customers SET photo_path=?, updated_at=? WHERE id=?",
            (photo_path, _now(), customer_id)
        )


def get_customer_photo(customer_id) -> str:
    """Return the stored photo_path for a customer ('' if none)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT photo_path FROM customers WHERE id=?", (customer_id,)
        ).fetchone()
    return str(row['photo_path'] or '') if row else ''


# ── Measurement CRUD ───────────────────────────────────────────────────────────

def save_measurement(customer_id, measurements: dict, height_cm=None,
                     garment_type='t_shirt', notes=''):
    now = _now()
    keys = ['shoulder_width', 'chest', 'waist', 'hip',
            'back_length', 'shirt_length', 'sleeve_length', 'neck']
    vals = [measurements.get(k) for k in keys]
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO measurements (customer_id, scan_date, "
            "shoulder_width, chest, waist, hip, back_length, shirt_length, "
            "sleeve_length, neck, height_cm, garment_type, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [customer_id, now] + vals + [height_cm, garment_type, notes]
        )
        # Update customer height
        conn.execute(
            'UPDATE customers SET updated_at=?, height_cm=COALESCE(?,height_cm) '
            'WHERE id=?',
            (now, height_cm, customer_id)
        )
        return cur.lastrowid


def get_measurements(customer_id, limit=20):
    with _connect() as conn:
        rows = conn.execute(
            'SELECT * FROM measurements WHERE customer_id=? '
            'ORDER BY scan_date DESC LIMIT ?',
            (customer_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]


def get_latest_measurement(customer_id):
    rows = get_measurements(customer_id, limit=1)
    return rows[0] if rows else None


# ── Order CRUD ─────────────────────────────────────────────────────────────────

ORDER_STATUSES = ['received', 'cutting', 'stitching', 'finishing', 'ready', 'delivered']
ORDER_STATUS_LABELS = {
    'received':  'Received',
    'cutting':   'Cutting',
    'stitching': 'Stitching',
    'finishing': 'Finishing',
    'ready':     'Ready',
    'delivered': 'Delivered',
}

def create_order(customer_id, garment_type, style_notes='', fabric='',
                 colour='', price=0.0, advance_paid=0.0,
                 delivery_date='', measurement_id=None, assigned_to=''):
    now = _now()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO orders (customer_id, measurement_id, garment_type, "
            "style_notes, fabric, colour, price, advance_paid, "
            "order_date, delivery_date, status, assigned_to) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,'received',?)",
            (customer_id, measurement_id, garment_type, style_notes,
             fabric, colour, price, advance_paid, now, delivery_date, assigned_to)
        )
        return cur.lastrowid


def update_order_status(order_id, status):
    if status not in ORDER_STATUSES:
        raise ValueError(f'Unknown status: {status}')
    with _connect() as conn:
        conn.execute('UPDATE orders SET status=? WHERE id=?', (status, order_id))


def get_orders(customer_id=None, status=None, limit=100):
    where, params = [], []
    if customer_id:
        where.append('o.customer_id=?'); params.append(customer_id)
    if status:
        where.append('o.status=?'); params.append(status)
    clause = ('WHERE ' + ' AND '.join(where)) if where else ''
    params.append(limit)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT o.*, c.name as customer_name, c.phone as customer_phone "
            f"FROM orders o JOIN customers c ON o.customer_id=c.id "
            f"{clause} ORDER BY o.order_date DESC LIMIT ?",
            params
        ).fetchall()
        return [dict(r) for r in rows]


def get_today_stats():
    today = datetime.date.today().isoformat()
    with _connect() as conn:
        scans  = conn.execute("SELECT COUNT(*) FROM measurements WHERE scan_date LIKE ?",
                              (f'{today}%',)).fetchone()[0]
        orders = conn.execute("SELECT COUNT(*) FROM orders WHERE order_date LIKE ?",
                              (f'{today}%',)).fetchone()[0]
        revenue = conn.execute("SELECT COALESCE(SUM(price),0) FROM orders WHERE order_date LIKE ?",
                               (f'{today}%',)).fetchone()[0]
        total_customers = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        active_orders   = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE status NOT IN ('delivered')"
        ).fetchone()[0]
        return {'scans': scans, 'orders': orders, 'revenue': revenue,
                'total_customers': total_customers, 'active_orders': active_orders}


# ── Inventory CRUD ─────────────────────────────────────────────────────────────

def get_inventory(category=None):
    """Return all inventory items, optionally filtered by category."""
    with _connect() as conn:
        if category:
            rows = conn.execute(
                "SELECT * FROM inventory WHERE category=? ORDER BY name",
                (category,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM inventory ORDER BY category, name"
            ).fetchall()
        return [dict(r) for r in rows]


def add_inventory_item(name, category='fabric', qty=0.0, unit='meters',
                       min_qty=2.0, colour='', price_per=0.0, notes=''):
    now = _now()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO inventory (name, category, qty, unit, min_qty, colour, "
            "price_per, notes, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (name, category, qty, unit, min_qty, colour, price_per, notes, now)
        )
        return cur.lastrowid


def update_inventory_qty(item_id, qty):
    """Set absolute quantity for an item."""
    with _connect() as conn:
        conn.execute(
            "UPDATE inventory SET qty=?, updated_at=? WHERE id=?",
            (qty, _now(), item_id)
        )


def adjust_inventory_qty(item_id, delta):
    """Add (positive) or subtract (negative) from quantity."""
    with _connect() as conn:
        conn.execute(
            "UPDATE inventory SET qty=MAX(0, qty+?), updated_at=? WHERE id=?",
            (delta, _now(), item_id)
        )


def delete_inventory_item(item_id):
    with _connect() as conn:
        conn.execute("DELETE FROM inventory WHERE id=?", (item_id,))


def get_low_stock_items():
    """Return items where qty < min_qty."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM inventory WHERE qty < min_qty ORDER BY (qty/MAX(min_qty,0.01))"
        ).fetchall()
        return [dict(r) for r in rows]


# ── Staff CRUD ─────────────────────────────────────────────────────────────────

def get_staff_list(active_only=True):
    """Return list of staff members."""
    with _connect() as conn:
        if active_only:
            rows = conn.execute(
                "SELECT * FROM staff_members WHERE active=1 ORDER BY name"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM staff_members ORDER BY active DESC, name"
            ).fetchall()
        return [dict(r) for r in rows]


def add_staff_member(name, role='Tailor'):
    now = _now()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO staff_members (name, role, active, created_at) VALUES (?,?,1,?)",
            (name, role, now)
        )
        return cur.lastrowid


def toggle_staff_active(staff_id):
    with _connect() as conn:
        conn.execute(
            "UPDATE staff_members SET active = 1 - active WHERE id=?",
            (staff_id,)
        )


# ── Loyalty / CRM ──────────────────────────────────────────────────────────────

def get_customer_loyalty(customer_id):
    """
    Calculate loyalty points for a customer.
    Rules: 1 point per 100 BDT spent + 10 points per completed order.
    Tiers: New (<100pts), Regular (100-499), Gold (500-999), VIP (1000+)
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as order_count, COALESCE(SUM(price),0) as total_spend "
            "FROM orders WHERE customer_id=?",
            (customer_id,)
        ).fetchone()
        if not row:
            return {'points': 0, 'tier': 'New', 'order_count': 0, 'total_spend': 0.0}

        order_count  = row['order_count']
        total_spend  = row['total_spend']
        points       = int(total_spend / 100) + (order_count * 10)

        if points >= 1000:
            tier = 'VIP'
        elif points >= 500:
            tier = 'Gold'
        elif points >= 100:
            tier = 'Regular'
        else:
            tier = 'New'

        return {
            'points':      points,
            'tier':        tier,
            'order_count': order_count,
            'total_spend': total_spend,
        }


# ── Tailor Feedback Loop ───────────────────────────────────────────────────────

FEEDBACK_FIELDS = [
    'shoulder_width', 'chest', 'waist', 'hip',
    'back_length', 'shirt_length', 'sleeve_length', 'neck',
]


def save_feedback(measurement_id: int, customer_id: int,
                  corrections: dict[str, tuple[float, float]],
                  tailor_note: str = ''):
    """
    Record tailor feedback for one measurement session.

    Parameters
    ----------
    measurement_id : int   — from measurements.id
    customer_id    : int   — from customers.id
    corrections    : {field_name: (measured_cm, actual_cm)}
        Only pass fields where the actual differs from what was measured.
        Example: {'chest': (96.5, 99.0), 'waist': (80.0, 78.5)}
    tailor_note    : str   — free-text note

    Stores one row per field in the feedback table.
    """
    now = _now()
    with _connect() as conn:
        for field, (measured, actual) in corrections.items():
            if field not in FEEDBACK_FIELDS:
                continue
            conn.execute(
                "INSERT INTO feedback "
                "(measurement_id, customer_id, feedback_date, field, "
                " measured_cm, actual_cm, tailor_note) "
                "VALUES (?,?,?,?,?,?,?)",
                (measurement_id, customer_id, now, field,
                 measured, actual, tailor_note)
            )


def get_feedback_corrections(min_samples: int = 3) -> dict[str, float]:
    """
    Compute the mean systematic error per field across all feedback.
    Returns {field: mean_error_cm} where error = actual - measured.
    Fields with fewer than min_samples feedbacks are excluded.

    Use this to shift prediction outputs: corrected = predicted + correction[field]
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT field, AVG(error_cm) as mean_err, COUNT(*) as n "
            "FROM feedback "
            "GROUP BY field "
            "HAVING n >= ?",
            (min_samples,)
        ).fetchall()
        return {r['field']: round(r['mean_err'], 2) for r in rows}


def get_feedback_stats() -> dict:
    """
    Return accuracy statistics from all tailor feedback.
    Includes per-field mean absolute error and total feedback count.
    """
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        rows = conn.execute(
            "SELECT field, "
            "       AVG(ABS(error_cm))  AS mae, "
            "       AVG(error_cm)       AS bias, "
            "       COUNT(*)            AS n "
            "FROM feedback "
            "GROUP BY field "
            "ORDER BY mae DESC"
        ).fetchall()
        fields = [dict(r) for r in rows]
    return {'total_feedback': total, 'by_field': fields}


def get_recent_feedback(customer_id: int = None, limit: int = 50) -> list[dict]:
    """Return recent feedback entries, optionally filtered by customer."""
    with _connect() as conn:
        if customer_id:
            rows = conn.execute(
                "SELECT f.*, c.name as customer_name, m.scan_date "
                "FROM feedback f "
                "JOIN customers c ON f.customer_id=c.id "
                "JOIN measurements m ON f.measurement_id=m.id "
                "WHERE f.customer_id=? "
                "ORDER BY f.feedback_date DESC LIMIT ?",
                (customer_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT f.*, c.name as customer_name, m.scan_date "
                "FROM feedback f "
                "JOIN customers c ON f.customer_id=c.id "
                "JOIN measurements m ON f.measurement_id=m.id "
                "ORDER BY f.feedback_date DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


# ── Migration helper ───────────────────────────────────────────────────────────

def migrate_json_records(json_path):
    """Import old records.json into the new database (one-time migration)."""
    if not os.path.exists(json_path):
        return 0
    try:
        with open(json_path) as f:
            records = json.load(f)
    except Exception:
        return 0
    count = 0
    for r in records:
        name = r.get('name', 'Unknown')
        cid = add_customer(name)
        m = {k: r.get(k) for k in
             ['shoulder_width','chest','waist','hip',
              'back_length','shirt_length','sleeve_length','neck']}
        save_measurement(cid, m)
        count += 1
    print(f'[DB] Migrated {count} records from {json_path}')
    return count


def _now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# Auto-init on import
init_db()
