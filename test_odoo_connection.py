import os
import xmlrpc.client
from dotenv import load_dotenv

load_dotenv()

ODOO_URL = os.getenv("ODOO_URL")
ODOO_DB = os.getenv("ODOO_DB")
ODOO_USERNAME = os.getenv("ODOO_USERNAME")
ODOO_PASSWORD = os.getenv("ODOO_PASSWORD")

common = xmlrpc.client.ServerProxy(f"{ODOO_URL}/xmlrpc/2/common")

uid = common.authenticate(
    ODOO_DB,
    ODOO_USERNAME,
    ODOO_PASSWORD,
    {}
)

if not uid:
    raise Exception("Odoo authentication failed. Check DB, username, or password.")

print("Connected to Odoo successfully")
print("UID:", uid)

models = xmlrpc.client.ServerProxy(f"{ODOO_URL}/xmlrpc/2/object")

version = common.version()
print("Odoo Version:", version)

partners = models.execute_kw(
    ODOO_DB,
    uid,
    ODOO_PASSWORD,
    "res.partner",
    "search_read",
    [[["id", "!=", False]]],
    {
        "fields": ["id", "name", "email", "phone"],
        "limit": 5
    }
)

print("Sample partners:")
for partner in partners:
    print(partner)
