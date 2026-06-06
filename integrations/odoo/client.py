import os
import xmlrpc.client
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


load_dotenv()


class OdooClient:
    def __init__(self):
        self.url = os.getenv("ODOO_URL", "http://localhost:8069")
        self.db = os.getenv("ODOO_DB")
        self.username = os.getenv("ODOO_USERNAME")
        self.password = os.getenv("ODOO_PASSWORD")

        if not self.db:
            raise ValueError("ODOO_DB is missing in .env")

        if not self.username:
            raise ValueError("ODOO_USERNAME is missing in .env")

        if not self.password:
            raise ValueError("ODOO_PASSWORD is missing in .env")

        self.common = xmlrpc.client.ServerProxy(
            f"{self.url}/xmlrpc/2/common",
            allow_none=True
        )

        self.models = xmlrpc.client.ServerProxy(
            f"{self.url}/xmlrpc/2/object",
            allow_none=True
        )

        self.uid = self._authenticate()

    def _authenticate(self) -> int:
        uid = self.common.authenticate(
            self.db,
            self.username,
            self.password,
            {}
        )

        if not uid:
            raise ConnectionError(
                "Odoo authentication failed. Check DB, username, or password."
            )

        return uid

    def version(self) -> Dict[str, Any]:
        return self.common.version()

    def search_read(
        self,
        model: str,
        domain: List,
        fields: Optional[List[str]] = None,
        limit: int = 10,
        order: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        kwargs: Dict[str, Any] = {
            "limit": limit,
        }

        if fields:
            kwargs["fields"] = fields

        if order:
            kwargs["order"] = order

        return self.models.execute_kw(
            self.db,
            self.uid,
            self.password,
            model,
            "search_read",
            [domain],
            kwargs
        )

    def read(
        self,
        model: str,
        ids: List[int],
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:

        kwargs: Dict[str, Any] = {}

        if fields:
            kwargs["fields"] = fields

        return self.models.execute_kw(
            self.db,
            self.uid,
            self.password,
            model,
            "read",
            [ids],
            kwargs
        )

    def create(
        self,
        model: str,
        values: Dict[str, Any],
    ) -> int:

        return self.models.execute_kw(
            self.db,
            self.uid,
            self.password,
            model,
            "create",
            [values]
        )

    def write(
        self,
        model: str,
        ids: List[int],
        values: Dict[str, Any],
    ) -> bool:

        return self.models.execute_kw(
            self.db,
            self.uid,
            self.password,
            model,
            "write",
            [ids, values]
        )

    def call_method(
        self,
        model: str,
        method: str,
        args: Optional[List] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:

        return self.models.execute_kw(
            self.db,
            self.uid,
            self.password,
            model,
            method,
            args or [],
            kwargs or {}
        )