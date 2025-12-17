"""Remote object handle for cross-process object references.

RemoteObjectHandle is a lightweight reference to an object living in another
process. It carries only the object_id and type_name, allowing the receiving
process to lazily fetch the actual object via RPC when needed.
"""
from __future__ import annotations


class RemoteObjectHandle:
    """Handle to an object in a remote process.

    This is a generic RPC concept - it represents a reference to an object
    that lives in another process. The actual object can be fetched via
    extension.get_remote_object(handle.object_id) on the receiving end.

    Attributes:
        object_id: Unique identifier for the remote object.
        type_name: The type name of the remote object (for debugging/logging).
    """

    # Preserve module identity for pickling compatibility
    __module__ = 'pyisolate._internal.remote_handle'

    def __init__(self, object_id: str, type_name: str) -> None:
        self.object_id = object_id
        self.type_name = type_name

    def __repr__(self) -> str:
        return f"<RemoteObject id={self.object_id} type={self.type_name}>"
